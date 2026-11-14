# One-command pipeline:  make all
# Each stage writes a sentinel in data/interim/.done/ so `make` reruns only what
# changed. Stages from L2 (provinces) onward also rerun when config.yaml or
# gazetteer.yaml change. Earlier, expensive stages (terrain, hydrology, features)
# do NOT track config.yaml; after changing their settings remove their sentinel,
# e.g. `rm data/interim/.done/hydro && make all`.
# `make clean` removes derived data (raw downloads are kept);
# `make distclean` also removes the raw cache.

PY      ?= python
CONFIG  ?= config.yaml
# mapped region: regions/$(REGION).yaml over config.yaml
REGION  ?= pak-afg
RUN      = $(PY) -m ecoregions --config $(CONFIG) --region $(REGION)
# stage markers in each region's own interim folder
DONE    := $(shell PYTHONPATH=src $(PY) -c "from ecoregions.config import load_config; print(load_config(None, '$(strip $(REGION))')['paths']['interim'])")/.done
SRC      = src/ecoregions

STAGES = download preflight grid dem terrain hydro covariates features provinces segment snap attributes communities vegcheck fauna faunal assemble evaluate audit validate crosscheck package join webmap staticmap

.PHONY: all clean distclean test names $(STAGES)
all: $(DONE)/staticmap $(DONE)/webmap $(DONE)/crosscheck $(DONE)/vegcheck $(DONE)/faunal $(DONE)/audit

$(DONE):
	mkdir -p $@

$(DONE)/download:                                                 | $(DONE) ; $(RUN) download   && touch $@
$(DONE)/preflight:  $(DONE)/download   $(SRC)/preflight.py $(CONFIG) regions/$(REGION).yaml | $(DONE) ; $(RUN) preflight  && touch $@
$(DONE)/grid:       $(DONE)/preflight  $(SRC)/boundaries.py       | $(DONE) ; $(RUN) grid       && touch $@
$(DONE)/dem:        $(DONE)/grid       $(SRC)/dem.py              | $(DONE) ; $(RUN) dem        && touch $@
$(DONE)/terrain:    $(DONE)/dem        $(SRC)/terrain.py          | $(DONE) ; $(RUN) terrain    && touch $@
$(DONE)/hydro:      $(DONE)/terrain    $(SRC)/hydro.py            | $(DONE) ; $(RUN) hydro      && touch $@
$(DONE)/covariates: $(DONE)/grid       $(SRC)/covariates.py       | $(DONE) ; $(RUN) covariates && touch $@
$(DONE)/features:   $(DONE)/hydro $(DONE)/covariates $(SRC)/features.py | $(DONE) ; $(RUN) features && touch $@
$(DONE)/provinces:  $(DONE)/features   $(SRC)/provinces.py $(CONFIG) regions/$(REGION).yaml | $(DONE) ; $(RUN) provinces && touch $@
$(DONE)/segment:    $(DONE)/provinces  $(SRC)/segment.py  $(CONFIG) regions/$(REGION).yaml | $(DONE) ; $(RUN) segment  && touch $@
$(DONE)/snap:       $(DONE)/segment    $(SRC)/snap.py   $(CONFIG) regions/$(REGION).yaml | $(DONE) ; $(RUN) snap       && touch $@
$(DONE)/attributes: $(DONE)/snap       $(SRC)/attributes.py gazetteer.yaml | $(DONE) ; $(RUN) attributes && touch $@
$(DONE)/communities: $(DONE)/attributes $(SRC)/communities.py communities.yaml | $(DONE) ; $(RUN) communities && touch $@
$(DONE)/vegcheck:   $(DONE)/communities $(SRC)/vegcheck.py $(CONFIG) regions/$(REGION).yaml | $(DONE) ; $(RUN) vegcheck   && touch $@
$(DONE)/fauna:      $(DONE)/vegcheck $(SRC)/fauna.py $(CONFIG) regions/$(REGION).yaml | $(DONE) ; $(RUN) fauna      && touch $@
$(DONE)/faunal:     $(DONE)/fauna $(SRC)/faunal.py $(CONFIG) regions/$(REGION).yaml | $(DONE) ; $(RUN) faunal     && touch $@
$(DONE)/assemble:   $(DONE)/faunal $(SRC)/assemble.py communities.yaml $(CONFIG) regions/$(REGION).yaml | $(DONE) ; $(RUN) assemble   && touch $@
$(DONE)/evaluate:   $(DONE)/assemble $(SRC)/evaluate.py $(CONFIG) regions/$(REGION).yaml | $(DONE) ; $(RUN) evaluate   && touch $@
$(DONE)/audit:      $(DONE)/evaluate $(SRC)/audit.py communities.yaml $(CONFIG) regions/$(REGION).yaml | $(DONE) ; $(RUN) audit      && touch $@
$(DONE)/validate:   $(DONE)/attributes $(SRC)/validate.py $(CONFIG) regions/$(REGION).yaml | $(DONE) ; $(RUN) validate   && touch $@
$(DONE)/crosscheck: $(DONE)/attributes $(SRC)/crosscheck.py gazetteer.yaml | $(DONE) ; $(RUN) crosscheck && touch $@
$(DONE)/package:    $(DONE)/validate $(DONE)/assemble $(SRC)/package.py          | $(DONE) ; $(RUN) package    && touch $@
$(DONE)/join:       $(DONE)/package    $(SRC)/join.py | $(DONE) ; $(RUN) join       && touch $@
$(DONE)/webmap:     $(DONE)/join       $(SRC)/webmap.py $(SRC)/colors.py $(CONFIG) regions/$(REGION).yaml | $(DONE) ; $(RUN) webmap     && touch $@
$(DONE)/staticmap:  $(DONE)/join $(DONE)/audit    $(SRC)/staticmap.py $(SRC)/colors.py $(CONFIG) regions/$(REGION).yaml | $(DONE) ; $(RUN) staticmap  && touch $@

# convenience aliases: `make snap` runs up to and including that stage
$(STAGES): %: $(DONE)/%

# rename/re-check units after editing gazetteer.yaml, without re-segmenting or re-vectorizing
names:
	$(RUN) names && $(RUN) crosscheck && $(RUN) package && $(RUN) join && $(RUN) webmap && $(RUN) staticmap

test:
	$(PY) -m pytest -q

clean:
	rm -rf data/interim outputs/qa

distclean: clean
	rm -rf data/raw whitebox_tools
