python -m venv venv

venv/bin/python -m pip install pip==24.0 setuptools==81.0.0
venv/bin/python -m pip install \
    --no-deps \
    --no-build-isolation \
    -r requirements-lock.txt

venv/bin/python -m pip install \
    --no-deps \
    -e submodules/perceptual-advex

# Installing dependencies independently
python -m pip install advex-uar recoloradv PyWavelets tensorboardX
python -m pip install git+https://github.com/MadryLab/robustness.git
python -m pip install git+https://github.com/fra31/auto-attack.git
python -m pip install -e submodules/perceptual-advex --no-deps