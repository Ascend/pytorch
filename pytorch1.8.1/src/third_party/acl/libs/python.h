

extern "C" {

class PyObject;

typedef  struct  _is
{
    struct _is *next;
    struct _ts *tstate_head;

    PyObject *modules;
    PyObject *sysdict;
    PyObject *builtins;
    PyObject *modules_reloading;

    PyObject *codec_search_path;
    PyObject *codec_search_cache;
    PyObject *codec_error_registry;

#ifdef HAVE_DLOPEN
    int dlopenflags;
#endif
#ifdef WITH_TSC
    int tscdump;
#endif

} PyInterpreterState;

typedef struct _PyThreadState
{
    PyInterpreterState *interp;
}PyThreadState;

int PyGILState_Check();
PyThreadState * PyEval_SaveThread();
void PyEval_RestoreThread(PyThreadState *tstate);

}