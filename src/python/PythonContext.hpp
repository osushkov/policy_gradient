#pragma once

#include <boost/python.hpp>
#include <chrono>
#include <iostream>
#include <mutex>
#include <thread>

namespace python {

struct PythonMainContext {
  PythonMainContext()
      : threadState(PyThreadState_Get()),
        interpreterState(threadState->interp) {
    PyEval_ReleaseLock();
  }

  ~PythonMainContext() {
    PyEval_RestoreThread(threadState);
    Py_Finalize();
  }

  PyThreadState *threadState;
  PyInterpreterState *interpreterState;

  std::mutex m;
};

struct PythonThreadContext {
  PythonThreadContext(PythonMainContext &mainCtx) : mainCtx(mainCtx) {
    PyEval_AcquireLock();
    threadState = PyThreadState_New(mainCtx.interpreterState);
    PyEval_ReleaseLock();
  }

  ~PythonThreadContext() {
    PyEval_AcquireLock();
    PyThreadState_Swap(mainCtx.threadState);
    PyThreadState_Clear(threadState);
    PyThreadState_Delete(threadState);
    PyEval_ReleaseLock();
  }

  PythonMainContext &mainCtx;
  PyThreadState *threadState;
};

struct PythonContextLock {
  PythonContextLock(PythonThreadContext &ctx) {
    // This is needed as a hack to prevent multiple threads from starving each
    // other on the global Python mutex. Seems to work well at least on the
    // Linux scheduler.
    std::this_thread::sleep_for(std::chrono::nanoseconds(1));
    lock = std::unique_lock<std::mutex>(ctx.mainCtx.m);

    PyEval_AcquireLock();
    prevThreadState = PyThreadState_Swap(ctx.threadState);
  }

  ~PythonContextLock() {
    PyThreadState_Swap(prevThreadState);
    PyEval_ReleaseLock();
  }

  PyThreadState *prevThreadState;
  std::unique_lock<std::mutex> lock;
};
}
