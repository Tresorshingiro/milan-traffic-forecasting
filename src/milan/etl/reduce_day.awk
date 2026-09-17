BEGIN {
    FS = "\t"
    OFMT = "%.9g"      # enough significant digits to survive a float32 round-trip
    CONVFMT = "%.9g"
    if (t0 == "") {
        print "ERROR: t0 not supplied (use -v t0=<epoch_ms>)" > "/dev/stderr"
        _error = 2
        exit 2
    }
}

{
    slot = int(($2 - t0) / 600000)
    if (slot < 0 || slot > 143) {
        printf "ERROR: slot %d out of range (ts=%s t0=%s) -- wrong date for this file?\n", \
               slot, $2, t0 > "/dev/stderr"
        _error = 3
        exit 3
    }
    if ($8 != "")
        g[($1 - 1) * 144 + slot] += $8
}

END {
    if (_error) exit _error
    for (s = 1; s <= 10000; s++) {
        base = (s - 1) * 144
        line = s
        for (j = 0; j < 144; j++) {
            k = base + j
            line = line " " (k in g ? g[k] : 0)
        }
        print line
    }
}