
# Need to set PYTHONPATH so that we pick up the local tstrait
PYPATH=${PWD}/..
TSTRAIT_VERSION:=$(shell PYTHONPATH=${PYPATH} \
   python3 -c 'import tstrait; print(tstrait.__version__.split("+")[0])')

dev:
	PYTHONPATH=${PYPATH} ./build.sh

dist:
	@echo Building distribution for tstrait version ${TSTRAIT_VERSION}
	sed -i s/__TSTRAIT_VERSION__/${TSTRAIT_VERSION}/g _config.yml
	PYTHONPATH=${PYPATH} ./build.sh

clean:
	rm -fR _build
