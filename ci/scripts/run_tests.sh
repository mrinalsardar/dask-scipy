#!/usr/bin/env bash

set -e

if [[ $PARALLEL == 'true' ]]; then
    export XTRATESTARGS="-n4 $XTRATESTARGS"
fi

if [[ $COVERAGE == 'true' ]]; then
    export XTRATESTARGS="--cov=dask --cov-report=xml $XTRATESTARGS"
fi

echo "py.test -v tests --runslow $XTRATESTARGS"
py.test -v tests --runslow $XTRATESTARGS

set +e
