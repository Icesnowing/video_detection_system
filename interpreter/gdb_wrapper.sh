#!/bin/bash
# GDB wrapper script to disable debuginfod
export DEBUGINFOD_URLS=""
exec /usr/bin/gdb "$@"
