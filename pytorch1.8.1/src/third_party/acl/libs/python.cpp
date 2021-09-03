#include "python.h"

int PyGILState_Check() {return 0;}
PyThreadState * PyEval_SaveThread() {return (PyThreadState*) 0;}
void PyEval_RestoreThread(PyThreadState *tstate) {return;}