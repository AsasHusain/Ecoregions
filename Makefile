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
RUN      = $(PY) -m ecoregions --config $(CONFIG)
DONE     = data/interim/.done
SRC      = src/ecoregions

STAGES = download grid dem terrain hydro covariates features provinces segment snap attributes communities vegcheck fauna faunal assemble evaluate validate crosscheck package webmap staticmap globe

.PHONY: all clean distclean test names $(STAGES)
all: $(DONE)/staticmap $(DONE)/webmap $(DONE)/crosscheck $(DONE)/vegcheck $(DONE)/faunal $(DONE)/evaluate $(DONE)/globe

$(DONE):
	mkdir -p $@

$(DONE)/download:                                                 | $(DONE) ; $(RUN) download   && touch $@
$(DONE)/grid:       $(DONE)/download   $(SRC)/boundaries.py       | $(DONE) ; $(RUN) grid       && touch $@
$(DONE)/dem:        $(DONE)/grid       $(SRC)/dem.py              | $(DONE) ; $(RUN) dem        && touch $@
$(DONE)/terrain:    $(DONE)/dem        $(SRC)/terrain.py          | $(DONE) ; $(RUN) terrain    && touch $@
$(DONE)/hydro:      $(DONE)/terrain    $(SRC)/hydro.py            | $(DONE) ; $(RUN) hydro      && touch $@
$(DONE)/covariates: $(DONE)/grid       $(SRC)/covariates.py       | $(DONE) ; $(RUN) covariates && touch $@
$(DONE)/features:   $(DONE)/hydro $(DONE)/covariates $(SRC)/features.py | $(DONE) ; $(RUN) features && touch $@
$(DONE)/provinces:  $(DONE)/features   $(SRC)/provinces.py $(CONFIG) | $(DONE) ; $(RUN) provinces && touch $@
$(DONE)/segment:    $(DONE)/provinces  $(SRC)/segment.py  $(CONFIG) | $(DONE) ; $(RUN) segment  && touch $@
$(DONE)/snap:       $(DONE)/segment    $(SRC)/snap.py   $(CONFIG) | $(DONE) ; $(RUN) snap       && touch $@
$(DONE)/attributes: $(DONE)/snap       $(SRC)/attributes.py gazetteer.yaml | $(DONE) ; $(RUN) attributes && touch $@
$(DONE)/communities: $(DONE)/attributes $(SRC)/communities.py communities.yaml | $(DONE) ; $(RUN) communities && touch $@
$(DONE)/vegcheck:   $(DONE)/communities $(SRC)/vegcheck.py $(CONFIG) | $(DONE) ; $(RUN) vegcheck   && touch $@
$(DONE)/fauna:      $(DONE)/vegcheck $(SRC)/fauna.py $(CONFIG) | $(DONE) ; $(RUN) fauna      && touch $@
$(DONE)/faunal:     $(DONE)/fauna $(SRC)/faunal.py $(CONFIG) | $(DONE) ; $(RUN) faunal     && touch $@
$(DONE)/assemble:   $(DONE)/faunal $(SRC)/assemble.py communities.yaml $(CONFIG) | $(DONE) ; $(RUN) assemble   && touch $@
$(DONE)/evaluate:   $(DONE)/assemble $(SRC)/evaluate.py $(CONFIG) | $(DONE) ; $(RUN) evaluate   && touch $@
$(DONE)/validate:   $(DONE)/attributes $(SRC)/validate.py $(CONFIG) | $(DONE) ; $(RUN) validate   && touch $@
$(DONE)/crosscheck: $(DONE)/attributes $(SRC)/crosscheck.py gazetteer.yaml | $(DONE) ; $(RUN) crosscheck && touch $@
$(DONE)/package:    $(DONE)/validate $(DONE)/assemble $(SRC)/package.py          | $(DONE) ; $(RUN) package    && touch $@
$(DONE)/webmap:     $(DONE)/package    $(SRC)/webmap.py $(SRC)/colors.py $(CONFIG) | $(DONE) ; $(RUN) webmap     && touch $@
$(DONE)/staticmap:  $(DONE)/package    $(SRC)/staticmap.py $(SRC)/colors.py $(CONFIG) | $(DONE) ; $(RUN) staticmap  && touch $@
$(DONE)/globe:      $(DONE)/package    $(SRC)/globe.py $(CONFIG) | $(DONE) ; $(RUN) globe      && touch $@

# convenience aliases: `make snap` runs up to and including that stage
$(STAGES): %: $(DONE)/%

# rename/re-check units after editing gazetteer.yaml, without re-segmenting or re-vectorizing
names:
	$(RUN) names && $(RUN) crosscheck && $(RUN) package && $(RUN) webmap && $(RUN) staticmap

test:
	$(PY) -m pytest -q

clean:
	rm -rf data/interim outputs/qa

distclean: clean
	rm -rf data/raw whitebox_tools
