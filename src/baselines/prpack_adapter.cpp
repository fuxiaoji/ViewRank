#include "prpack_base_graph.h"
#include "prpack_solver.h"
#include "prpack_result.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <vector>

using namespace prpack;
using Clock = std::chrono::steady_clock;
static double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now()-start).count();
}
namespace prpack {
struct prpack_adapter_access {
    static prpack_preprocessed_scc_graph* scc(prpack_solver* solver) { return solver->sccg; }
    static prpack_preprocessed_gs_graph* gs(prpack_solver* solver) { return solver->gsg; }
    static prpack_preprocessed_scc_graph* prepare_scc(prpack_solver* solver) {
        if (!solver->sccg) solver->sccg=new prpack_preprocessed_scc_graph(solver->bg);
        return solver->sccg;
    }
    static prpack_result* linear_scc(prpack_solver* solver,double alpha,double tol,const double* rhs) {
        auto* p=solver->sccg;
        // Original residual-based kernel, without probability normalization.
        return solver->solve_via_scc_gs(alpha,tol,p->num_vs,
            p->num_es_inside,p->heads_inside,p->tails_inside,p->vals_inside,
            p->num_es_outside,p->heads_outside,p->tails_outside,p->vals_outside,
            p->ii,p->d,p->num_outlinks,rhs,p->num_comps,p->divisions,
            p->encoding,p->decoding,false);
    }
};
}
struct NativePRPack {
    prpack_solver* solver; prpack_base_graph* graph; int n; int nnz;
    std::vector<int> locations;
    std::vector<unsigned char> kinds;
    bool use_scc;
};

static void map_weights(NativePRPack* h) {
    if (!h->locations.empty()) return;
    h->locations.resize(h->nnz);h->kinds.resize(h->nnz);
    if (!h->use_scc) {
        int cursor=0;
        for(int target=0;target<h->n;++target) {
            int stop=target+1<h->n?h->graph->tails[target+1]:h->nnz;
            for(int j=h->graph->tails[target];j<stop;++j) {
                bool self=h->graph->heads[j]==target;
                h->kinds[j]=self?0:1;h->locations[j]=self?target:cursor++;
            }
        }
        return;
    }
    auto* p=prpack_adapter_access::scc(h->solver);
    for (int comp=0;comp<p->num_comps;++comp) {
        int begin=p->divisions[comp],end=comp+1<p->num_comps?p->divisions[comp+1]:h->n;
        for (int i=begin;i<end;++i) {
            int target=p->decoding[i];
            int inside=p->tails_inside[i],outside=p->tails_outside[i];
            int stop=target+1<h->n?h->graph->tails[target+1]:h->nnz;
            for (int j=h->graph->tails[target];j<stop;++j) {
                int source=p->encoding[h->graph->heads[j]];
                if (source==i) {h->kinds[j]=0;h->locations[j]=i;}
                else if (begin<=source&&source<end) {h->kinds[j]=1;h->locations[j]=inside++;}
                else {h->kinds[j]=2;h->locations[j]=outside++;}
            }
        }
    }
}

extern "C" __declspec(dllexport) void* prpack_create_csr(
        int n, int nnz, const int* indptr, const int* indices,
        const double* values, int use_scc, double* preparation_seconds) {
    auto start = Clock::now();
    auto* bg = new prpack_base_graph();
    bg->num_vs=n; bg->num_es=nnz; bg->num_self_es=0;
    bg->tails=new int[n]; bg->heads=new int[nnz]; bg->vals=new double[nnz];
    std::copy(indptr,indptr+n,bg->tails);
    std::copy(indices,indices+nnz,bg->heads);
    std::copy(values,values+nnz,bg->vals);
    for (int i=0;i<n;++i)
        for (int j=indptr[i];j<indptr[i+1];++j)
            if (indices[j]==i) ++bg->num_self_es;
    // Input CSR is P[target,source], hence exactly PRPACK's in-link layout.
    // Values already obey source-column normalization; no semantic change.
    auto* handle=new NativePRPack{new prpack_solver(bg,true),bg,n,nnz,{}, {},use_scc!=0};
    *preparation_seconds=seconds(start);
    return handle;
}

extern "C" __declspec(dllexport) int prpack_solve_personalized(
        void* pointer, const double* seed, double alpha, double tolerance,
        double* output, double* statistics) {
    auto* handle=static_cast<NativePRPack*>(pointer);
    auto start=Clock::now();
    // Upstream auto selection uses SCCGS for n>=128; all study graphs qualify.
    auto* result=handle->solver->solve(alpha,tolerance,seed,seed,handle->use_scc?"sccgs":"gs");
    std::copy(result->x,result->x+handle->n,output);
    statistics[1]=result->preprocess_time;
    statistics[2]=result->compute_time;
    statistics[3]=static_cast<double>(result->num_es_touched);
    // Upstream SCC-GS does not initialize result->converged. Never read it.
    const bool mapping_missing=handle->locations.empty();
    auto mapping_started=Clock::now();
    map_weights(handle);
    statistics[4]=mapping_missing?seconds(mapping_started):0;
    statistics[0]=seconds(start);
    delete result;
    return 0;
}

extern "C" __declspec(dllexport) void prpack_update_weights(
        void* pointer,int count,const int* positions,const double* weights) {
    auto* h=static_cast<NativePRPack*>(pointer);
    auto* p=prpack_adapter_access::scc(h->solver);
    auto* g=prpack_adapter_access::gs(h->solver);
    for(int i=0;i<count;++i) {
        int j=positions[i];double difference=weights[i]-h->graph->vals[j];
        h->graph->vals[j]=weights[i];
        if (!p&&!g) continue;
        int location=h->locations[j];
        if(g) {
            if(h->kinds[j]==0) g->ii[location]=weights[i];
            else g->vals[location]=weights[i];
            g->d[h->graph->heads[j]]-=difference;
            continue;
        }
        if(h->kinds[j]==0) p->ii[location]=weights[i];
        else if(h->kinds[j]==1) p->vals_inside[location]=weights[i];
        else p->vals_outside[location]=weights[i];
        p->d[p->encoding[h->graph->heads[j]]]-=difference;
    }
}

extern "C" __declspec(dllexport) int prpack_solve_linear_scc(
        void* pointer,const double* rhs,double alpha,double tolerance,
        double* output,double* statistics) {
    auto* h=static_cast<NativePRPack*>(pointer);
    if (!h->use_scc) return 1;
    auto start=Clock::now();
    const bool missing=prpack_adapter_access::scc(h->solver)==nullptr;
    auto prep_started=Clock::now();
    prpack_adapter_access::prepare_scc(h->solver);
    statistics[1]=missing?seconds(prep_started):0;
    auto compute_started=Clock::now();
    auto* result=prpack_adapter_access::linear_scc(h->solver,alpha,tolerance,rhs);
    statistics[2]=seconds(compute_started);
    std::copy(result->x,result->x+h->n,output);
    statistics[3]=static_cast<double>(result->num_es_touched);
    const bool mapping_missing=h->locations.empty();
    auto mapping_started=Clock::now();
    map_weights(h);
    statistics[4]=mapping_missing?seconds(mapping_started):0;
    statistics[0]=seconds(start);
    delete result;
    return 0;
}

extern "C" __declspec(dllexport) void prpack_memory(
        void* pointer,double* statistics) {
    auto* h=static_cast<NativePRPack*>(pointer);
    auto* p=prpack_adapter_access::scc(h->solver);
    statistics[0]=12.0*h->nnz+4.0*h->n;
    auto* g=prpack_adapter_access::gs(h->solver);
    statistics[1]=p?(24.0*p->num_es+32.0*h->n+4.0*p->num_comps):(g?12.0*g->num_es+20.0*h->n:0);
    statistics[2]=5.0*h->locations.size();
    statistics[3]=p?p->num_comps:0;
}

extern "C" __declspec(dllexport) void prpack_destroy(void* pointer) {
    auto* handle=static_cast<NativePRPack*>(pointer);
    delete handle->solver;
    delete handle;
}
