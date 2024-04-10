# Building the Docs

We use sphinx to build the documentation.  To build, make sure you have the sphinx dependencies installed, then run:

```
sphinx-build -b html source/. _build/html
```

This will output the generated html into the `_build/html` directory which you can then open in your browser.
