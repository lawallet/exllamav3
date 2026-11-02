"""Opt-in one-second telemetry, separate from primary timing. No torch import."""
import argparse, ctypes as C, json, os, subprocess, threading, time
from pathlib import Path
import psutil

class Value(C.Structure):
    _fields_=[('status',C.c_ulong),('value',C.c_double)]
class Item(C.Structure):
    _fields_=[('name',C.c_wchar_p),('value',Value)]

class PDH:
    def __init__(self):
        self.dll=C.WinDLL('pdh');self.query=C.c_void_p();self.handles={};self.errors={}
        d=self.dll
        d.PdhOpenQueryW.argtypes=[C.c_wchar_p,C.c_size_t,C.POINTER(C.c_void_p)]
        d.PdhAddEnglishCounterW.argtypes=[C.c_void_p,C.c_wchar_p,C.c_size_t,C.POINTER(C.c_void_p)]
        d.PdhCollectQueryData.argtypes=[C.c_void_p]
        d.PdhGetFormattedCounterArrayW.argtypes=[C.c_void_p,C.c_ulong,C.POINTER(C.c_ulong),C.POINTER(C.c_ulong),C.c_void_p]
        d.PdhCloseQuery.argtypes=[C.c_void_p]
        assert d.PdhOpenQueryW(None,0,C.byref(self.query))==0
        paths=[r'\Processor Information(*)\% Processor Time',r'\Processor Information(*)\% Processor Performance',
            r'\Processor Information(*)\Processor Frequency',r'\System\Processor Queue Length',r'\System\Context Switches/sec',
            r'\Memory\Available MBytes',r'\Memory\Committed Bytes',r'\Memory\Page Faults/sec',r'\Memory\Page Reads/sec',
            r'\Memory\Pages Input/sec',r'\Memory\Cache Bytes',r'\Memory\System Cache Resident Bytes',
            r'\PhysicalDisk(*)\Disk Read Bytes/sec',r'\PhysicalDisk(*)\Disk Reads/sec',r'\PhysicalDisk(*)\Avg. Disk sec/Read',
            r'\GPU Adapter Memory(*)\Dedicated Usage',r'\GPU Adapter Memory(*)\Shared Usage']
        for path in paths:
            h=C.c_void_p();err=d.PdhAddEnglishCounterW(self.query,path,0,C.byref(h))
            if err:self.errors[path]=err
            else:self.handles[path]=h
        d.PdhCollectQueryData(self.query)
    def sample(self):
        self.dll.PdhCollectQueryData(self.query);out={}
        for path,h in self.handles.items():
            size=C.c_ulong();count=C.c_ulong()
            self.dll.PdhGetFormattedCounterArrayW(h,0x200|0x8000,C.byref(size),C.byref(count),None)
            if not size.value:continue
            buf=C.create_string_buffer(size.value)
            err=self.dll.PdhGetFormattedCounterArrayW(h,0x200|0x8000,C.byref(size),C.byref(count),buf)
            if err:out[path]={'error':err};continue
            arr=C.cast(buf,C.POINTER(Item))
            out[path]={arr[i].name:arr[i].value.value for i in range(count.value) if arr[i].value.status in (0,1)}
        return out

class PCIe:
    def __init__(self):
        self.error=None;self.handles={}
        try:
            self.dll=C.WinDLL('nvml.dll');d=self.dll
            if d.nvmlInit_v2():raise RuntimeError('nvmlInit_v2 failed')
            d.nvmlDeviceGetHandleByIndex_v2.argtypes=[C.c_uint,C.POINTER(C.c_void_p)]
            d.nvmlDeviceGetPcieThroughput.argtypes=[C.c_void_p,C.c_uint,C.POINTER(C.c_uint)]
            for i in (0,2):
                h=C.c_void_p();err=d.nvmlDeviceGetHandleByIndex_v2(i,C.byref(h))
                if not err:self.handles[i]=h
        except Exception as e:self.error=str(e)
    def sample(self):
        out={'error':self.error} if self.error else {}
        for i,h in self.handles.items():
            out[i]={}
            for name,kind in [('tx_kib_s',0),('rx_kib_s',1)]:
                value=C.c_uint();err=self.dll.nvmlDeviceGetPcieThroughput(h,kind,C.byref(value))
                out[i][name]=dict(error=err) if err else value.value
        return out

def groups(process):
    k=C.WinDLL('kernel32',use_last_error=True)
    k.OpenProcess.argtypes=[C.c_ulong,C.c_int,C.c_ulong];k.OpenProcess.restype=C.c_void_p
    k.CloseHandle.argtypes=[C.c_void_p]
    k.GetProcessGroupAffinity.argtypes=[C.c_void_p,C.POINTER(C.c_ushort),C.POINTER(C.c_ushort)]
    h=k.OpenProcess(0x400,False,process.pid);n=C.c_ushort(64);g=(C.c_ushort*64)()
    try:
        if not k.GetProcessGroupAffinity(h,C.byref(n),g):return {'error':C.get_last_error()}
        return list(g[:n.value])
    finally:
        if h:k.CloseHandle(h)

def thread_groups(process):
    class Affinity(C.Structure):
        _fields_=[('mask',C.c_size_t),('group',C.c_ushort),('reserved',C.c_ushort*3)]
    k=C.WinDLL('kernel32',use_last_error=True);k.OpenThread.argtypes=[C.c_ulong,C.c_int,C.c_ulong];k.OpenThread.restype=C.c_void_p
    k.CloseHandle.argtypes=[C.c_void_p];k.GetThreadGroupAffinity.argtypes=[C.c_void_p,C.POINTER(Affinity)]
    result={}
    for t in process.threads():
        h=k.OpenThread(0x40,False,t.id);a=Affinity()
        try:
            if h and k.GetThreadGroupAffinity(h,C.byref(a)):result[t.id]=dict(group=a.group,mask=hex(a.mask))
        finally:
            if h:k.CloseHandle(h)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--pid',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--exact-pid',action='store_true',help='Provided PID is inference, not a venv launcher')
    p.add_argument('--seconds',type=int,default=3600);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    pdh=PDH();pcie=PCIe();launcher=psutil.Process(a.pid)
    candidates=[launcher]+[q for q in launcher.children(recursive=True) if q.name().lower()=='python.exe']
    process=launcher if a.exact_pid else max(candidates,key=lambda q:q.memory_info().rss)
    monitor_lineage={os.getpid(),*[q.pid for q in psutil.Process().parents() if q.pid!=process.pid]}
    tree_pids=[process.pid]+[q.pid for q in process.children(recursive=True) if q.pid not in monitor_lineage]
    fields='timestamp,index,clocks.sm,clocks.mem,power.draw,utilization.gpu,utilization.memory,pstate,temperature.gpu,memory.used,memory.free,pcie.link.gen.current,pcie.link.width.current,clocks_event_reasons.active'
    gpu=subprocess.Popen(['nvidia-smi','-i','0,2','--query-gpu='+fields,'--format=csv','-lms','1000'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,creationflags=subprocess.CREATE_NO_WINDOW)
    def read_gpu():
        with (a.output/'gpu.jsonl').open('w') as f:
            for line in gpu.stdout:f.write(json.dumps(dict(time_ns=time.time_ns(),csv=line.strip()))+'\n');f.flush()
    threading.Thread(target=read_gpu,daemon=True).start()
    (a.output/'metadata.json').write_text(json.dumps(dict(pid=process.pid,launcher_pid=a.pid,process_tree_pids=tree_pids,start_ns=time.time_ns(),pdh_errors=pdh.errors,
        cpu_count=psutil.cpu_count(),affinity=process.cpu_affinity(),processor_groups=groups(process),thread_groups=thread_groups(process),cpu_freq=psutil.cpu_freq()._asdict(),
        note='PDH performance is frequency-relative performance, not per-thread effective cycle residency; IO bytes include cache reads.'),indent=2))
    try:
        with (a.output/'host.jsonl').open('w') as f:
            for _ in range(a.seconds):
                time.sleep(1)
                if (a.output/'stop').exists() or not process.is_running():break
                row=dict(time_ns=time.time_ns(),pdh=pdh.sample(),pcie=pcie.sample(),system=psutil.virtual_memory()._asdict(),swap=psutil.swap_memory()._asdict())
                with process.oneshot():
                    row['process']=dict(cpu_times=process.cpu_times()._asdict(),memory=process.memory_info()._asdict(),
                        io=process.io_counters()._asdict(),threads=[t._asdict() for t in process.threads()],
                        ctx=process.num_ctx_switches()._asdict(),thread_count=process.num_threads())
                row['other_cpu']=[dict(pid=q.pid,name=q.info['name'],cpu=q.info['cpu_times'][:2])
                    for q in psutil.process_iter(['name','cpu_times']) if q.info['cpu_times']]
                f.write(json.dumps(row)+'\n');f.flush()
    except psutil.NoSuchProcess:
        pass
    finally:
        gpu.terminate();gpu.wait(timeout=10);pdh.dll.PdhCloseQuery(pdh.query)

if __name__=='__main__':main()
