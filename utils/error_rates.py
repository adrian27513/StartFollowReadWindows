import editdistance
def cer(r, h):
    #Remove any double or trailing
    r = ' '.join(r.split())
    h = ' '.join(h.split())
    # r = str(bytes(r, 'ansi'), 'utf-8')
    # h = str(bytes(r, 'ansi'), 'utf-8')
    # print("R:", r)
    # print("H:", h)
    # if h == "":
    #     quit()
    return err(r, h)

def err(r, h):

    dis = editdistance.eval(r, h)
    # print("Dis", dis)
    # if len(r) == 0.0:
    #     return len(h)
    # print("Num", float(dis))
    # print("Den", float(len(r)))
    # print("CER:", float(dis) / float(len(r)))
    return float(dis) / float(len(r))

def wer(r, h):
    r = r.split()
    h = h.split()

    return err(r,h)
