import time
import threading
import cv2


def operation1():
    start = time.time()
    cv2.imread("D:\\Intern\\Kafka\\test_images\\1.jpg")
    end = time.time()
    print("Operation 1 completed in %.4f seconds" % (end - start))


def operation2():
    start = time.time()
    cv2.imread("D:\\Intern\\Kafka\\test_images\\2.jpg")
    end = time.time()
    print("Operation 2 completed in %.4f seconds" % (end - start))


def operation3():
    start = time.time()
    cv2.imread("D:\\Intern\\Kafka\\test_images\\3.jpg")
    end = time.time()
    print("Operation 3 completed in %.4f seconds" % (end - start))


def operation4():
    start = time.time()
    cv2.imread("D:\\Intern\\Kafka\\test_images\\4.jpg")
    end = time.time()
    print("Operation 4 completed in %.4f seconds" % (end - start))


def asyn_method():  #异步多线程方式
    print("Asyn method started.")
    start_time = time.time()

    thread1 = threading.Thread(target=operation1)  #首先定义多线程
    thread2 = threading.Thread(target=operation2)
    thread3 = threading.Thread(target=operation3)
    thread4 = threading.Thread(target=operation4)

    thread1.start()  #其次运行线程，相当于快速按下线程开关，线程启动后继续异步向下执行下面代码
    thread2.start()
    thread3.start()
    thread4.start()

    thread1.join()  #最后汇总，等待线程1完成
    thread2.join()
    thread3.join()
    thread4.join()

    end_time = time.time()
    print("Asyn method ended.")
    print("Total time: %.4f seconds" % (end_time - start_time))


def syn_method():
    print("Syn method started.")
    start_time = time.time()
    operation1()
    operation2()
    operation3()
    operation4()
    end_time = time.time()
    print("Syn method ended.")
    print("Total time: %.4f seconds" % (end_time - start_time))    


if __name__ == "__main__":
    syn_method()
    print()
    asyn_method()  #异步多线程函数
