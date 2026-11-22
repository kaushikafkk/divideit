# Security policy

## Supported versions

DivideIT is an early MVP. Security fixes are applied to the latest release and the `main` branch.

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub's security-advisory feature instead of a
public issue. Include a minimal reproduction, affected operating system, and model metadata when
relevant. Do not attach proprietary model weights.

## Current safety boundaries

Version 0.1 reads untrusted GGUF metadata but does not execute model code. The parser validates
string and array sizes before allocation. Future runtime-controller releases will treat model paths,
runtime arguments, and child-process output as untrusted input.

