"""Read-only per-process WDDM budgets, using SDK dxgi.h/dxgi1_4.h interfaces."""
import ctypes as C,uuid
class GUID(C.Structure):_fields_=[('data',C.c_ubyte*16)]
def guid(text):return GUID((C.c_ubyte*16).from_buffer_copy(uuid.UUID(text).bytes_le))
class LUID(C.Structure):_fields_=[('low',C.c_ulong),('high',C.c_long)]
class Desc(C.Structure):
    _fields_=[('description',C.c_wchar*128),('vendor',C.c_uint),('device',C.c_uint),('subsys',C.c_uint),('revision',C.c_uint),
        ('dedicated',C.c_size_t),('system',C.c_size_t),('shared',C.c_size_t),('luid',LUID),('flags',C.c_uint)]
class Info(C.Structure):
    _fields_=[('budget',C.c_ulonglong),('usage',C.c_ulonglong),('available_reservation',C.c_ulonglong),('reservation',C.c_ulonglong)]
def method(ptr,index,result,*args):
    table=C.cast(ptr,C.POINTER(C.POINTER(C.c_void_p))).contents
    return C.WINFUNCTYPE(result,C.c_void_p,*args)(table[index])
def release(ptr):
    if ptr:method(ptr,2,C.c_ulong)(ptr)
def query():
    factory=C.c_void_p();d=C.WinDLL('dxgi');iid=guid('770aae78-f26f-4dba-a829-253c83d1b387')
    d.CreateDXGIFactory1.argtypes=[C.POINTER(GUID),C.POINTER(C.c_void_p)];d.CreateDXGIFactory1.restype=C.c_long
    error=d.CreateDXGIFactory1(C.byref(iid),C.byref(factory))
    if error:return {'error':error}
    result=[]
    try:
        enum=method(factory,12,C.c_long,C.c_uint,C.POINTER(C.c_void_p))
        for index in range(16):
            adapter=C.c_void_p();adapter3=C.c_void_p()
            error=enum(factory,index,C.byref(adapter))
            if error:break
            try:
                desc=Desc();error=method(adapter,10,C.c_long,C.POINTER(Desc))(adapter,C.byref(desc))
                if error:continue
                iid3=guid('645967a4-1392-4310-a798-8053ce3e93fd')
                error=method(adapter,0,C.c_long,C.POINTER(GUID),C.POINTER(C.c_void_p))(adapter,C.byref(iid3),C.byref(adapter3))
                if error:continue
                row=dict(index=index,name=desc.description,dedicated=desc.dedicated,luid=f'{desc.luid.high & 0xffffffff:08x}:{desc.luid.low:08x}')
                for kind,name in ((0,'local'),(1,'nonlocal')):
                    info=Info();error=method(adapter3,14,C.c_long,C.c_uint,C.c_uint,C.POINTER(Info))(adapter3,0,kind,C.byref(info))
                    row[name]={'error':error} if error else {k:getattr(info,k) for k,_ in Info._fields_}
                result.append(row)
            finally:release(adapter3);release(adapter)
    finally:release(factory)
    return result
if __name__=='__main__':
    import json
    print(json.dumps(query(),indent=2))
