

def test(a:int,b:int,hook = None):

    if hook:
        hook(a,b)
    else:
        print(a-b)

def hook(a,b):
    print(a+b)

test(1,2,hook)