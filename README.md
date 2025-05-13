# gnuradio-examples
Examples for GNURadio documentation I am working on.

Notes:
To display in Visual studio code
export DISPLAY=localhost:10.0


To decode the Embedded Python Block: File Source to Tagged Stream by Barry Duggan. A bash one‑liner that will strip any number of leading %UU…U] chunks (i.e. % + one‑or‑more U + ]) and then hand off whatever’s left to base64 --decode:
```
 sed -E 's/^(%U+\])+'// "output.tmp" | base64 --decode
```