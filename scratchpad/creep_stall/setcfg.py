"""params.yaml 의 standoff 스위치/값을 임시로 바꿔 리플레이한다 (검증 전용).
반드시 restore() 로 되돌린다 — git checkout -- 로도 복구 가능."""
import io, re, sys
P='config/params.yaml'
def setk(k, v):
    s=io.open(P,encoding='utf-8').read()
    pat=re.compile(r'^(  %s: ).*$' % re.escape(k), re.M)
    assert pat.search(s), k
    s=pat.sub(lambda m: m.group(1)+v, s)
    io.open(P,'w',encoding='utf-8').write(s)
if __name__=='__main__':
    for a in sys.argv[1:]:
        k,v=a.split('='); setk(k,v)
    print('set', sys.argv[1:])
