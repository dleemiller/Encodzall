#include <Python.h>
#include <string.h>

#include "constants.h"


PyObject* tokenize(PyObject* self, PyObject* args, PyObject* kwargs) {
    char* text;
    PyObject* max_length_obj = Py_None;
    int pad_id = DEFAULT_PAD;
    int end_id = DEFAULT_END;
    static char* kwlist[] = {"text", "max_length", "pad_id", "end_id", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "s|Oii", kwlist, &text, &max_length_obj, &pad_id, &end_id)) {
        return NULL;
    }

    long max_length = -1;
    if (max_length_obj != Py_None) {
        max_length = PyLong_AsLong(max_length_obj);
        if (PyErr_Occurred()) return NULL; // Error converting max_length
    }

    size_t text_len = strlen(text);

    PyObject* tokens = PyList_New(0);
    PyObject* attention_mask = PyList_New(0);
    PyObject* word_starts = PyList_New(0);

    if (!tokens || !attention_mask || !word_starts) {
        Py_XDECREF(tokens);
        Py_XDECREF(attention_mask);
        Py_XDECREF(word_starts);
        return NULL;
    }

    int last_was_space = 1;
    for (size_t i = 0; i < text_len; ++i) {
        if (max_length > 0 && PyList_Size(tokens) >= max_length - 1) break; // Reserve space for end_id

        unsigned char ch = text[i];
        int is_space = isspace(ch);

        PyList_Append(tokens, PyLong_FromLong(ch));
        PyList_Append(attention_mask, PyBool_FromLong(1));
        PyList_Append(word_starts, PyBool_FromLong(last_was_space && !is_space));
        
        last_was_space = is_space;
    }

    // Add end_id if there's space
    if (max_length <= 0 || PyList_Size(tokens) < max_length) {
        PyList_Append(tokens, PyLong_FromLong(end_id));
        PyList_Append(attention_mask, PyBool_FromLong(1));
        PyList_Append(word_starts, PyBool_FromLong(0));
    }

    // Padding if necessary
    while (max_length > 0 && PyList_Size(tokens) < max_length) {
        PyList_Append(tokens, PyLong_FromLong(pad_id));
        PyList_Append(attention_mask, PyBool_FromLong(0));
        PyList_Append(word_starts, PyBool_FromLong(0));
    }

    return PyTuple_Pack(3, tokens, attention_mask, word_starts);
}

