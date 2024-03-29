#include <Python.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

PyObject *AugmentationFailed;

#include "tokenizer.h"
#include "constants.h"


// Method definitions
static PyMethodDef EncodZallMethods[] = {
    {"tokenize", (PyCFunction)tokenize, METH_VARARGS | METH_KEYWORDS, "Tokenize a string."},
    {"make_targets", (PyCFunction)make_targets, METH_VARARGS | METH_KEYWORDS, "Segment words from tokenized string."},
    {NULL, NULL, 0, NULL} // Sentinel
};

// Module definition
static struct PyModuleDef encodzallmodule = {
    PyModuleDef_HEAD_INIT,
    "encodzall",
    "Module for encodzall.",
    -1,
    EncodZallMethods
};

// Module initialization
PyMODINIT_FUNC PyInit_tokenizer(void) {
    PyObject *module = PyModule_Create(&encodzallmodule);
    if (module == NULL) {
        return NULL;
    }

    PyModule_AddIntConstant(module, "DEFAULT_PAD_ID", DEFAULT_PAD);
    PyModule_AddIntConstant(module, "DEFAULT_END_ID", DEFAULT_END);
    PyModule_AddIntConstant(module, "DEFAULT_TARGET_PAD_ID", DEFAULT_TARGET_PAD);
    PyModule_AddIntConstant(module, "DEFAULT_TARGET_END_ID", DEFAULT_TARGET_END);


    return module;
}


