import threading
from time import sleep

def threadedFunc(arg):
    for i in range(arg):
        print("Run!")

if __name__ == "__main__":
    thread = threading.Thread(target = threadedFunc, args = (10, ))
    thread.start()
    thread.join()
    print("finished")