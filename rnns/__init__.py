# coding: utf-8

from rnns import gru, lstm, atr, sru, lrn, olrn


def get_cell(cell_name, hidden_size, ln=False, scope=None):
    """Convert the cell_name into cell instance."""
    cell_name = cell_name.lower()

    if cell_name == "gru":
        return gru.gru(hidden_size, ln=ln, scope=scope or "gru")
    elif cell_name == "lstm":
        return lstm.lstm(hidden_size, ln=ln, scope=scope or "lstm")
    elif cell_name == "atr":
        return atr.atr(hidden_size, ln=ln, scope=scope or "atr")
    elif cell_name == "sru":
        return sru.sru(hidden_size, ln=ln, scope=scope or "sru")
    elif cell_name == "lrn":
        return lrn.lrn(hidden_size, ln=ln, scope=scope or "lrn")
    elif cell_name == "olrn":
        return olrn.olrn(hidden_size, ln=ln, scope=scope or "olrn")
    else:
        raise NotImplementedError("{} is not supported".format(cell_name))
