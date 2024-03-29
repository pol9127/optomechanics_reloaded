/*
  From Multipack project
 */
#include "__quadpack.h"

static struct PyMethodDef quadpack_module_methods[] = {
{"_qagse", quadpack_qagse, METH_VARARGS, doc_qagse},
{NULL,		NULL, 0, NULL}
};

#if PY_VERSION_HEX >= 0x03000000
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_quadpack",
    NULL,
    -1,
    quadpack_module_methods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyObject *PyInit__quadpack(void)
{
    PyObject *m, *d, *s;

    m = PyModule_Create(&moduledef);
    import_array();
    d = PyModule_GetDict(m);

    s = PyUnicode_FromString(" 1.13 ");
    PyDict_SetItemString(d, "__version__", s);
    quadpack_error = PyErr_NewException ("quadpack.error", NULL, NULL);
    Py_DECREF(s);
    PyDict_SetItemString(d, "error", quadpack_error);
    if (PyErr_Occurred()) {
        Py_FatalError("can't initialize module quadpack");
    }
    return m;
}
#else
PyMODINIT_FUNC init_quadpack(void) {
  PyObject *m, *d, *s;
  m = Py_InitModule("_quadpack", quadpack_module_methods);
  import_array();
  d = PyModule_GetDict(m);

  s = PyString_FromString(" 1.13 ");
  PyDict_SetItemString(d, "__version__", s);
  quadpack_error = PyErr_NewException ("quadpack.error", NULL, NULL);
  Py_DECREF(s);
  PyDict_SetItemString(d, "error", quadpack_error);
  if (PyErr_Occurred())
    Py_FatalError("can't initialize module quadpack");
}
#endif
