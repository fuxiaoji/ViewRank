"""Windows process working-set counters, including native allocations."""
import ctypes as ct
from ctypes import wintypes


class _Counters(ct.Structure):
    _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[
        (name,ct.c_size_t) for name in ['PeakWorkingSetSize','WorkingSetSize',
        'QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage',
        'QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage','PrivateUsage']]


def process_memory():
    counters=_Counters();counters.cb=ct.sizeof(counters)
    api=ct.WinDLL('psapi').GetProcessMemoryInfo
    api.argtypes=[wintypes.HANDLE,ct.POINTER(_Counters),wintypes.DWORD]
    api.restype=wintypes.BOOL
    if not api(wintypes.HANDLE(-1),ct.byref(counters),counters.cb):raise ct.WinError()
    return dict(process_working_set_bytes=counters.WorkingSetSize,
                process_peak_working_set_bytes=counters.PeakWorkingSetSize,
                process_private_commit_bytes=counters.PrivateUsage)
