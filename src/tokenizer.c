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

void adjust_to_max_words(PyObject* words_list, PyObject* word_attention_list, int max_words, int max_length, int pad_id) {
    size_t current_words = PyList_Size(words_list);

    // Pad words_list and word_attention_list to max_words
    if (max_words > 0 && current_words < max_words) {
        PyObject* pad_list = PyList_New(max_length);
        for (int i = 0; i < max_length; ++i) {
            PyList_SetItem(pad_list, i, PyLong_FromLong(pad_id)); // Reference stolen, no need to Py_DECREF
        }
        while (PyList_Size(words_list) < max_words) {
            PyList_Append(words_list, pad_list);
            PyList_Append(word_attention_list, Py_False);
        }
        Py_DECREF(pad_list); // Cleanup
    }

    // Truncate words_list and word_attention_list to max_words
    while (max_words > 0 && PyList_Size(words_list) > max_words) {
        PySequence_DelItem(words_list, PyList_Size(words_list) - 1);
        PySequence_DelItem(word_attention_list, PyList_Size(word_attention_list) - 1);
    }
}


void finalize_and_append_word(PyObject* current_word, PyObject* words_list, int max_length, int pad_id, int end_id, int current_word_has_non_pad, PyObject* word_attention_list) {
    // Append end_id if there's room or if max_length is not set
    if (max_length < 0 || PyList_Size(current_word) < max_length) {
        PyList_Append(current_word, PyLong_FromLong(end_id));
    }

    // Pad or truncate current_word to max_length
    while (max_length > 0 && PyList_Size(current_word) < max_length) {
        PyList_Append(current_word, PyLong_FromLong(pad_id));
    }
    while (max_length > 0 && PyList_Size(current_word) > max_length) {
        PySequence_DelItem(current_word, PyList_Size(current_word) - 1); // Remove last item
    }

    // Append the finalized word to the words list
    PyList_Append(words_list, current_word);

    // Update the word attention list
    PyObject* attention_value = current_word_has_non_pad ? Py_True : Py_False;
    Py_INCREF(attention_value); // Because PyList_Append steals a reference
    PyList_Append(word_attention_list, attention_value);
}


PyObject* segment_words(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject* tokens;
    PyObject* attention_mask;
    PyObject* word_starts;
    int max_length = -1;
    int max_words = -1;
    int pad_id = DEFAULT_TARGET_PAD;
    int end_id = DEFAULT_TARGET_END;
    static char* kwlist[] = {"tokens", "attention_mask", "word_starts", "max_length", "max_words", "pad_id", "end_id", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO|iiii", kwlist, 
                                     &tokens, &attention_mask, &word_starts, 
                                     &max_length, &max_words, &pad_id, &end_id)) {
        return NULL; // Error parsing arguments
    }

    // Validate input lists
    if (!PyList_Check(tokens) || !PyList_Check(attention_mask) || !PyList_Check(word_starts)) {
        PyErr_SetString(PyExc_TypeError, "All inputs must be lists");
        return NULL;
    }
    
    size_t tokens_len = PyList_Size(tokens);
    size_t attention_mask_len = PyList_Size(attention_mask);
    size_t word_starts_len = PyList_Size(word_starts);
    
    if (tokens_len != attention_mask_len || tokens_len != word_starts_len) {
        PyErr_SetString(PyExc_ValueError, "All lists must have the same length");
        return NULL;
    }


    PyObject* words_list = PyList_New(0);
    PyObject* word_attention_list = PyList_New(0);
    PyObject* current_word = PyList_New(0);
    int current_word_has_non_pad = 0;
    size_t token_count = PyList_Size(tokens);

    for (size_t i = 0; i < token_count; ++i) {
        PyObject* token = PyList_GetItem(tokens, i);
        PyObject* is_word_start = PyList_GetItem(word_starts, i);
        PyObject* is_attention = PyList_GetItem(attention_mask, i);

        if (PyObject_IsTrue(is_word_start) && PyList_Size(current_word) > 0) {
            // Finish the current word, apply end_id, pad or truncate, then start a new word
            finalize_and_append_word(current_word, words_list, max_length, pad_id, end_id, current_word_has_non_pad, word_attention_list);
            current_word = PyList_New(0); // Start a new word
            current_word_has_non_pad = 0;
        }

        if (PyObject_IsTrue(is_attention)) {
            current_word_has_non_pad = 1;
        }

        // Append token to current word
        PyList_Append(current_word, token);

        // Handle last word
        if (i == token_count - 1) {
            finalize_and_append_word(current_word, words_list, max_length, pad_id, end_id, current_word_has_non_pad, word_attention_list);
        }
    }

    // Pad or truncate the words_list and word_attention_list to max_words
    adjust_to_max_words(words_list, word_attention_list, max_words, max_length, pad_id);

    return Py_BuildValue("OO", words_list, word_attention_list);
}

