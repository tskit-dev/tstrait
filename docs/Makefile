
# Need to set PYTHONPATH so that we pick up the local tstrait
PYPATH=${PWD}/..
MSP_VERSION:=$(shell PYTHONPATH=${PYPATH} \
   python3 -c 'import tstrait')

dev:
	PYTHONPATH=${PYPATH} ./build.sh

clean:
	rm -fR _build
