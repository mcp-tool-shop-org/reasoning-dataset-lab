# Pilot B Reader Think-Arm Comparison (nemotron)
Note: In Ollama's nemotron3nano renderer (`model/renderers/nemotron3nano.go:30, 86-87`), `"medium"` is an effort hint the template knows (thinking stays on), not a cap on thinking tokens.

## Summary
- Reader family: `nemotron`
- Parent think setting: `True`
- Addendum think setting: `'medium'`
- Total sets: 50
- Total directed pairs: 1000
- Parent length stops (reported separately from verdicts; never counted as a verdict or folded into disagree): 25
- Addendum length stops (reported separately from verdicts; never counted as a verdict or folded into disagree): 22
- Pairs with a verdict at both settings: 954
- Exact verdict agreement count: 911
- Verdict disagreement count (verdict pairs only): 43
- Exact verdict agreement rate (over all pairs): 911 / 1000 (91.1%)
- Exact verdict agreement rate (over pairs with a verdict at both settings): 911 / 954 (95.5%)
- Per-set would pass (nemotron half only, parent `True`): 29 / 50
- Per-set would pass (nemotron half only, addendum `'medium'`): 30 / 50

## Cross-Table (Parent Rows vs Addendum Columns; `length` Separated from Verdicts)
| Parent \ Addendum | incompatible (verdict) | compatible (verdict) | unsure (verdict) | length (stop — not a verdict) |
| --- | ---: | ---: | ---: | ---: |
| incompatible (verdict) | 862 | 20 | 2 | 11 |
| compatible (verdict) | 15 | 49 | 1 | 8 |
| unsure (verdict) | 4 | 1 | 0 | 1 |
| length (stop — not a verdict) | 9 | 11 | 3 | 2 |

## Per-Set Would-Pass by Setting (nemotron half only)
| Set | Parent Would Pass (nemotron half) | Addendum Would Pass (nemotron half) | Parent Length Stops | Addendum Length Stops |
| --- | :---: | :---: | ---: | ---: |
| turbines/turbine mode | fail | fail | 0 | 1 |
| turbines/casing alloy | pass | pass | 0 | 0 |
| turbines/cooling medium | pass | pass | 0 | 0 |
| turbines/lubricant grade | pass | pass | 0 | 0 |
| turbines/governor mode | fail | fail | 1 | 3 |
| relays/breaker position | fail | fail | 1 | 2 |
| relays/contact plating | pass | pass | 0 | 0 |
| relays/enclosure rating | fail | fail | 3 | 2 |
| relays/mount style | pass | pass | 0 | 0 |
| relays/coil voltage | pass | pass | 0 | 0 |
| vessels/berth status | pass | pass | 0 | 0 |
| vessels/topcoat | pass | pass | 0 | 0 |
| vessels/registry port | pass | pass | 0 | 0 |
| vessels/ice class | pass | pass | 0 | 0 |
| vessels/propulsion type | pass | pass | 0 | 0 |
| furnaces/damper position | fail | fail | 4 | 2 |
| furnaces/lining brick | pass | pass | 0 | 0 |
| furnaces/purge gas | pass | pass | 0 | 0 |
| furnaces/thermocouple type | fail | pass | 1 | 0 |
| furnaces/hearth drive | fail | fail | 0 | 1 |
| reactors/loop state | fail | fail | 1 | 2 |
| reactors/cladding metal | pass | pass | 0 | 0 |
| reactors/moderator type | pass | pass | 0 | 0 |
| reactors/shielding material | pass | pass | 0 | 0 |
| reactors/coolant type | pass | pass | 0 | 0 |
| compressors/compressor mode | pass | fail | 0 | 1 |
| compressors/rotor coating | fail | pass | 1 | 0 |
| compressors/intercooler medium | pass | pass | 0 | 0 |
| compressors/shaft seal | pass | pass | 0 | 0 |
| compressors/unloader mechanism | fail | fail | 2 | 0 |
| transformers/tap position | fail | fail | 3 | 2 |
| transformers/core steel | fail | fail | 1 | 2 |
| transformers/dielectric fluid | fail | fail | 0 | 2 |
| transformers/bushing material | pass | pass | 0 | 0 |
| transformers/vector group | pass | pass | 0 | 0 |
| boilers/burner state | pass | pass | 0 | 0 |
| boilers/tube alloy | pass | pass | 0 | 0 |
| boilers/feedwater program | fail | fail | 1 | 0 |
| boilers/sootblower type | fail | fail | 2 | 0 |
| boilers/firing method | fail | fail | 0 | 0 |
| chillers/circuit state | pass | pass | 0 | 0 |
| chillers/brazing filler | pass | pass | 0 | 0 |
| chillers/refrigerant | pass | pass | 0 | 0 |
| chillers/expansion valve | fail | fail | 2 | 0 |
| chillers/condenser cooling | pass | pass | 0 | 0 |
| pumps/pump state | fail | fail | 0 | 0 |
| pumps/seal material | pass | pass | 0 | 0 |
| pumps/drive coupling | fail | fail | 1 | 1 |
| pumps/bearing type | fail | fail | 0 | 0 |
| pumps/casing mount | fail | fail | 1 | 1 |

## Per-Pair Verdicts and Length Stops
| Set | First | Second | Parent | Addendum | Classification |
| --- | --- | --- | --- | --- | --- |
| turbines/turbine mode | standby | barring | compatible | incompatible | verdict_disagree |
| turbines/turbine mode | standby | starting | incompatible | incompatible | verdict_agree |
| turbines/turbine mode | standby | online | incompatible | incompatible | verdict_agree |
| turbines/turbine mode | standby | coasting | incompatible | incompatible | verdict_agree |
| turbines/turbine mode | barring | standby | compatible | compatible | verdict_agree |
| turbines/turbine mode | barring | starting | compatible | incompatible | verdict_disagree |
| turbines/turbine mode | barring | online | incompatible | incompatible | verdict_agree |
| turbines/turbine mode | barring | coasting | incompatible | incompatible | verdict_agree |
| turbines/turbine mode | starting | standby | incompatible | incompatible | verdict_agree |
| turbines/turbine mode | starting | barring | unsure | compatible | verdict_disagree |
| turbines/turbine mode | starting | online | compatible | incompatible | verdict_disagree |
| turbines/turbine mode | starting | coasting | incompatible | incompatible | verdict_agree |
| turbines/turbine mode | online | standby | incompatible | incompatible | verdict_agree |
| turbines/turbine mode | online | barring | incompatible | incompatible | verdict_agree |
| turbines/turbine mode | online | starting | incompatible | incompatible | verdict_agree |
| turbines/turbine mode | online | coasting | incompatible | length | length_stop_excluded_from_verdicts |
| turbines/turbine mode | coasting | standby | incompatible | incompatible | verdict_agree |
| turbines/turbine mode | coasting | barring | incompatible | incompatible | verdict_agree |
| turbines/turbine mode | coasting | starting | incompatible | incompatible | verdict_agree |
| turbines/turbine mode | coasting | online | incompatible | compatible | verdict_disagree |
| turbines/casing alloy | steel | castiron | incompatible | incompatible | verdict_agree |
| turbines/casing alloy | steel | titanium | incompatible | incompatible | verdict_agree |
| turbines/casing alloy | steel | inconel | incompatible | incompatible | verdict_agree |
| turbines/casing alloy | steel | aluminum | incompatible | incompatible | verdict_agree |
| turbines/casing alloy | castiron | steel | incompatible | incompatible | verdict_agree |
| turbines/casing alloy | castiron | titanium | incompatible | incompatible | verdict_agree |
| turbines/casing alloy | castiron | inconel | incompatible | incompatible | verdict_agree |
| turbines/casing alloy | castiron | aluminum | incompatible | incompatible | verdict_agree |
| turbines/casing alloy | titanium | steel | incompatible | incompatible | verdict_agree |
| turbines/casing alloy | titanium | castiron | incompatible | incompatible | verdict_agree |
| turbines/casing alloy | titanium | inconel | incompatible | incompatible | verdict_agree |
| turbines/casing alloy | titanium | aluminum | incompatible | incompatible | verdict_agree |
| turbines/casing alloy | inconel | steel | incompatible | incompatible | verdict_agree |
| turbines/casing alloy | inconel | castiron | incompatible | incompatible | verdict_agree |
| turbines/casing alloy | inconel | titanium | incompatible | incompatible | verdict_agree |
| turbines/casing alloy | inconel | aluminum | incompatible | incompatible | verdict_agree |
| turbines/casing alloy | aluminum | steel | incompatible | incompatible | verdict_agree |
| turbines/casing alloy | aluminum | castiron | incompatible | incompatible | verdict_agree |
| turbines/casing alloy | aluminum | titanium | incompatible | incompatible | verdict_agree |
| turbines/casing alloy | aluminum | inconel | incompatible | incompatible | verdict_agree |
| turbines/cooling medium | helium | nitrogen | incompatible | incompatible | verdict_agree |
| turbines/cooling medium | helium | hydrogen | incompatible | incompatible | verdict_agree |
| turbines/cooling medium | helium | water | incompatible | incompatible | verdict_agree |
| turbines/cooling medium | helium | oil | incompatible | incompatible | verdict_agree |
| turbines/cooling medium | nitrogen | helium | incompatible | incompatible | verdict_agree |
| turbines/cooling medium | nitrogen | hydrogen | incompatible | incompatible | verdict_agree |
| turbines/cooling medium | nitrogen | water | incompatible | incompatible | verdict_agree |
| turbines/cooling medium | nitrogen | oil | incompatible | incompatible | verdict_agree |
| turbines/cooling medium | hydrogen | helium | incompatible | incompatible | verdict_agree |
| turbines/cooling medium | hydrogen | nitrogen | incompatible | incompatible | verdict_agree |
| turbines/cooling medium | hydrogen | water | incompatible | incompatible | verdict_agree |
| turbines/cooling medium | hydrogen | oil | incompatible | incompatible | verdict_agree |
| turbines/cooling medium | water | helium | incompatible | incompatible | verdict_agree |
| turbines/cooling medium | water | nitrogen | incompatible | incompatible | verdict_agree |
| turbines/cooling medium | water | hydrogen | incompatible | incompatible | verdict_agree |
| turbines/cooling medium | water | oil | incompatible | incompatible | verdict_agree |
| turbines/cooling medium | oil | helium | incompatible | incompatible | verdict_agree |
| turbines/cooling medium | oil | nitrogen | incompatible | incompatible | verdict_agree |
| turbines/cooling medium | oil | hydrogen | incompatible | incompatible | verdict_agree |
| turbines/cooling medium | oil | water | incompatible | incompatible | verdict_agree |
| turbines/lubricant grade | mineral | polyalphaolefin | incompatible | incompatible | verdict_agree |
| turbines/lubricant grade | mineral | ester | incompatible | incompatible | verdict_agree |
| turbines/lubricant grade | mineral | polyglycol | incompatible | incompatible | verdict_agree |
| turbines/lubricant grade | mineral | silicone | incompatible | incompatible | verdict_agree |
| turbines/lubricant grade | polyalphaolefin | mineral | incompatible | incompatible | verdict_agree |
| turbines/lubricant grade | polyalphaolefin | ester | incompatible | incompatible | verdict_agree |
| turbines/lubricant grade | polyalphaolefin | polyglycol | incompatible | incompatible | verdict_agree |
| turbines/lubricant grade | polyalphaolefin | silicone | incompatible | incompatible | verdict_agree |
| turbines/lubricant grade | ester | mineral | incompatible | incompatible | verdict_agree |
| turbines/lubricant grade | ester | polyalphaolefin | incompatible | incompatible | verdict_agree |
| turbines/lubricant grade | ester | polyglycol | incompatible | incompatible | verdict_agree |
| turbines/lubricant grade | ester | silicone | incompatible | incompatible | verdict_agree |
| turbines/lubricant grade | polyglycol | mineral | incompatible | incompatible | verdict_agree |
| turbines/lubricant grade | polyglycol | polyalphaolefin | incompatible | incompatible | verdict_agree |
| turbines/lubricant grade | polyglycol | ester | incompatible | incompatible | verdict_agree |
| turbines/lubricant grade | polyglycol | silicone | incompatible | incompatible | verdict_agree |
| turbines/lubricant grade | silicone | mineral | incompatible | incompatible | verdict_agree |
| turbines/lubricant grade | silicone | polyalphaolefin | incompatible | incompatible | verdict_agree |
| turbines/lubricant grade | silicone | ester | incompatible | incompatible | verdict_agree |
| turbines/lubricant grade | silicone | polyglycol | incompatible | incompatible | verdict_agree |
| turbines/governor mode | droop | isochronous | incompatible | incompatible | verdict_agree |
| turbines/governor mode | droop | baseload | compatible | incompatible | verdict_disagree |
| turbines/governor mode | droop | pressure | incompatible | incompatible | verdict_agree |
| turbines/governor mode | droop | loadlimit | incompatible | incompatible | verdict_agree |
| turbines/governor mode | isochronous | droop | incompatible | incompatible | verdict_agree |
| turbines/governor mode | isochronous | baseload | length | compatible | length_stop_excluded_from_verdicts |
| turbines/governor mode | isochronous | pressure | incompatible | incompatible | verdict_agree |
| turbines/governor mode | isochronous | loadlimit | incompatible | incompatible | verdict_agree |
| turbines/governor mode | baseload | droop | incompatible | length | length_stop_excluded_from_verdicts |
| turbines/governor mode | baseload | isochronous | compatible | length | length_stop_excluded_from_verdicts |
| turbines/governor mode | baseload | pressure | incompatible | incompatible | verdict_agree |
| turbines/governor mode | baseload | loadlimit | incompatible | incompatible | verdict_agree |
| turbines/governor mode | pressure | droop | incompatible | incompatible | verdict_agree |
| turbines/governor mode | pressure | isochronous | incompatible | length | length_stop_excluded_from_verdicts |
| turbines/governor mode | pressure | baseload | incompatible | incompatible | verdict_agree |
| turbines/governor mode | pressure | loadlimit | incompatible | incompatible | verdict_agree |
| turbines/governor mode | loadlimit | droop | incompatible | incompatible | verdict_agree |
| turbines/governor mode | loadlimit | isochronous | incompatible | incompatible | verdict_agree |
| turbines/governor mode | loadlimit | baseload | incompatible | incompatible | verdict_agree |
| turbines/governor mode | loadlimit | pressure | incompatible | incompatible | verdict_agree |
| relays/breaker position | service | test | incompatible | incompatible | verdict_agree |
| relays/breaker position | service | disconnected | incompatible | incompatible | verdict_agree |
| relays/breaker position | service | withdrawn | incompatible | incompatible | verdict_agree |
| relays/breaker position | service | earthed | incompatible | incompatible | verdict_agree |
| relays/breaker position | test | service | incompatible | incompatible | verdict_agree |
| relays/breaker position | test | disconnected | unsure | incompatible | verdict_disagree |
| relays/breaker position | test | withdrawn | incompatible | incompatible | verdict_agree |
| relays/breaker position | test | earthed | incompatible | length | length_stop_excluded_from_verdicts |
| relays/breaker position | disconnected | service | incompatible | incompatible | verdict_agree |
| relays/breaker position | disconnected | test | incompatible | incompatible | verdict_agree |
| relays/breaker position | disconnected | withdrawn | incompatible | compatible | verdict_disagree |
| relays/breaker position | disconnected | earthed | incompatible | compatible | verdict_disagree |
| relays/breaker position | withdrawn | service | incompatible | incompatible | verdict_agree |
| relays/breaker position | withdrawn | test | incompatible | incompatible | verdict_agree |
| relays/breaker position | withdrawn | disconnected | compatible | compatible | verdict_agree |
| relays/breaker position | withdrawn | earthed | length | compatible | length_stop_excluded_from_verdicts |
| relays/breaker position | earthed | service | incompatible | incompatible | verdict_agree |
| relays/breaker position | earthed | test | compatible | length | length_stop_excluded_from_verdicts |
| relays/breaker position | earthed | disconnected | incompatible | compatible | verdict_disagree |
| relays/breaker position | earthed | withdrawn | compatible | compatible | verdict_agree |
| relays/contact plating | silver | gold | incompatible | incompatible | verdict_agree |
| relays/contact plating | silver | palladium | incompatible | incompatible | verdict_agree |
| relays/contact plating | silver | rhodium | incompatible | incompatible | verdict_agree |
| relays/contact plating | silver | platinum | incompatible | incompatible | verdict_agree |
| relays/contact plating | gold | silver | incompatible | incompatible | verdict_agree |
| relays/contact plating | gold | palladium | incompatible | incompatible | verdict_agree |
| relays/contact plating | gold | rhodium | incompatible | incompatible | verdict_agree |
| relays/contact plating | gold | platinum | incompatible | incompatible | verdict_agree |
| relays/contact plating | palladium | silver | incompatible | incompatible | verdict_agree |
| relays/contact plating | palladium | gold | incompatible | incompatible | verdict_agree |
| relays/contact plating | palladium | rhodium | incompatible | incompatible | verdict_agree |
| relays/contact plating | palladium | platinum | incompatible | incompatible | verdict_agree |
| relays/contact plating | rhodium | silver | incompatible | incompatible | verdict_agree |
| relays/contact plating | rhodium | gold | incompatible | incompatible | verdict_agree |
| relays/contact plating | rhodium | palladium | incompatible | incompatible | verdict_agree |
| relays/contact plating | rhodium | platinum | incompatible | incompatible | verdict_agree |
| relays/contact plating | platinum | silver | incompatible | incompatible | verdict_agree |
| relays/contact plating | platinum | gold | incompatible | incompatible | verdict_agree |
| relays/contact plating | platinum | palladium | incompatible | incompatible | verdict_agree |
| relays/contact plating | platinum | rhodium | incompatible | incompatible | verdict_agree |
| relays/enclosure rating | nema3r | nema4 | incompatible | incompatible | verdict_agree |
| relays/enclosure rating | nema3r | nema7 | incompatible | incompatible | verdict_agree |
| relays/enclosure rating | nema3r | nema12 | incompatible | compatible | verdict_disagree |
| relays/enclosure rating | nema3r | nema13 | incompatible | incompatible | verdict_agree |
| relays/enclosure rating | nema4 | nema3r | incompatible | incompatible | verdict_agree |
| relays/enclosure rating | nema4 | nema7 | incompatible | incompatible | verdict_agree |
| relays/enclosure rating | nema4 | nema12 | length | compatible | length_stop_excluded_from_verdicts |
| relays/enclosure rating | nema4 | nema13 | incompatible | incompatible | verdict_agree |
| relays/enclosure rating | nema7 | nema3r | incompatible | compatible | verdict_disagree |
| relays/enclosure rating | nema7 | nema4 | incompatible | incompatible | verdict_agree |
| relays/enclosure rating | nema7 | nema12 | incompatible | incompatible | verdict_agree |
| relays/enclosure rating | nema7 | nema13 | incompatible | length | length_stop_excluded_from_verdicts |
| relays/enclosure rating | nema12 | nema3r | length | incompatible | length_stop_excluded_from_verdicts |
| relays/enclosure rating | nema12 | nema4 | incompatible | length | length_stop_excluded_from_verdicts |
| relays/enclosure rating | nema12 | nema7 | incompatible | incompatible | verdict_agree |
| relays/enclosure rating | nema12 | nema13 | compatible | incompatible | verdict_disagree |
| relays/enclosure rating | nema13 | nema3r | incompatible | compatible | verdict_disagree |
| relays/enclosure rating | nema13 | nema4 | length | incompatible | length_stop_excluded_from_verdicts |
| relays/enclosure rating | nema13 | nema7 | incompatible | incompatible | verdict_agree |
| relays/enclosure rating | nema13 | nema12 | incompatible | incompatible | verdict_agree |
| relays/mount style | rack | rail | incompatible | incompatible | verdict_agree |
| relays/mount style | rack | wall | incompatible | incompatible | verdict_agree |
| relays/mount style | rack | pole | incompatible | incompatible | verdict_agree |
| relays/mount style | rack | floor | incompatible | incompatible | verdict_agree |
| relays/mount style | rail | rack | incompatible | incompatible | verdict_agree |
| relays/mount style | rail | wall | incompatible | incompatible | verdict_agree |
| relays/mount style | rail | pole | incompatible | incompatible | verdict_agree |
| relays/mount style | rail | floor | incompatible | incompatible | verdict_agree |
| relays/mount style | wall | rack | incompatible | incompatible | verdict_agree |
| relays/mount style | wall | rail | incompatible | incompatible | verdict_agree |
| relays/mount style | wall | pole | incompatible | incompatible | verdict_agree |
| relays/mount style | wall | floor | incompatible | incompatible | verdict_agree |
| relays/mount style | pole | rack | incompatible | incompatible | verdict_agree |
| relays/mount style | pole | rail | incompatible | incompatible | verdict_agree |
| relays/mount style | pole | wall | incompatible | incompatible | verdict_agree |
| relays/mount style | pole | floor | incompatible | incompatible | verdict_agree |
| relays/mount style | floor | rack | incompatible | incompatible | verdict_agree |
| relays/mount style | floor | rail | incompatible | incompatible | verdict_agree |
| relays/mount style | floor | wall | incompatible | incompatible | verdict_agree |
| relays/mount style | floor | pole | incompatible | incompatible | verdict_agree |
| relays/coil voltage | dc24 | dc48 | incompatible | incompatible | verdict_agree |
| relays/coil voltage | dc24 | dc125 | incompatible | incompatible | verdict_agree |
| relays/coil voltage | dc24 | ac120 | incompatible | incompatible | verdict_agree |
| relays/coil voltage | dc24 | ac240 | incompatible | incompatible | verdict_agree |
| relays/coil voltage | dc48 | dc24 | incompatible | incompatible | verdict_agree |
| relays/coil voltage | dc48 | dc125 | incompatible | incompatible | verdict_agree |
| relays/coil voltage | dc48 | ac120 | incompatible | incompatible | verdict_agree |
| relays/coil voltage | dc48 | ac240 | incompatible | incompatible | verdict_agree |
| relays/coil voltage | dc125 | dc24 | incompatible | incompatible | verdict_agree |
| relays/coil voltage | dc125 | dc48 | incompatible | incompatible | verdict_agree |
| relays/coil voltage | dc125 | ac120 | incompatible | incompatible | verdict_agree |
| relays/coil voltage | dc125 | ac240 | incompatible | incompatible | verdict_agree |
| relays/coil voltage | ac120 | dc24 | incompatible | incompatible | verdict_agree |
| relays/coil voltage | ac120 | dc48 | incompatible | incompatible | verdict_agree |
| relays/coil voltage | ac120 | dc125 | incompatible | incompatible | verdict_agree |
| relays/coil voltage | ac120 | ac240 | incompatible | incompatible | verdict_agree |
| relays/coil voltage | ac240 | dc24 | incompatible | incompatible | verdict_agree |
| relays/coil voltage | ac240 | dc48 | incompatible | incompatible | verdict_agree |
| relays/coil voltage | ac240 | dc125 | incompatible | incompatible | verdict_agree |
| relays/coil voltage | ac240 | ac120 | incompatible | incompatible | verdict_agree |
| vessels/berth status | underway | anchored | incompatible | incompatible | verdict_agree |
| vessels/berth status | underway | alongside | incompatible | incompatible | verdict_agree |
| vessels/berth status | underway | adrift | incompatible | incompatible | verdict_agree |
| vessels/berth status | underway | aground | incompatible | incompatible | verdict_agree |
| vessels/berth status | anchored | underway | incompatible | incompatible | verdict_agree |
| vessels/berth status | anchored | alongside | incompatible | incompatible | verdict_agree |
| vessels/berth status | anchored | adrift | incompatible | incompatible | verdict_agree |
| vessels/berth status | anchored | aground | incompatible | incompatible | verdict_agree |
| vessels/berth status | alongside | underway | incompatible | incompatible | verdict_agree |
| vessels/berth status | alongside | anchored | incompatible | incompatible | verdict_agree |
| vessels/berth status | alongside | adrift | incompatible | incompatible | verdict_agree |
| vessels/berth status | alongside | aground | incompatible | incompatible | verdict_agree |
| vessels/berth status | adrift | underway | incompatible | incompatible | verdict_agree |
| vessels/berth status | adrift | anchored | incompatible | incompatible | verdict_agree |
| vessels/berth status | adrift | alongside | incompatible | incompatible | verdict_agree |
| vessels/berth status | adrift | aground | incompatible | incompatible | verdict_agree |
| vessels/berth status | aground | underway | incompatible | incompatible | verdict_agree |
| vessels/berth status | aground | anchored | incompatible | incompatible | verdict_agree |
| vessels/berth status | aground | alongside | incompatible | incompatible | verdict_agree |
| vessels/berth status | aground | adrift | incompatible | incompatible | verdict_agree |
| vessels/topcoat | epoxy | polyurethane | incompatible | incompatible | verdict_agree |
| vessels/topcoat | epoxy | silicone | incompatible | incompatible | verdict_agree |
| vessels/topcoat | epoxy | acrylic | incompatible | incompatible | verdict_agree |
| vessels/topcoat | epoxy | vinyl | incompatible | incompatible | verdict_agree |
| vessels/topcoat | polyurethane | epoxy | incompatible | incompatible | verdict_agree |
| vessels/topcoat | polyurethane | silicone | incompatible | incompatible | verdict_agree |
| vessels/topcoat | polyurethane | acrylic | incompatible | incompatible | verdict_agree |
| vessels/topcoat | polyurethane | vinyl | incompatible | incompatible | verdict_agree |
| vessels/topcoat | silicone | epoxy | incompatible | incompatible | verdict_agree |
| vessels/topcoat | silicone | polyurethane | incompatible | incompatible | verdict_agree |
| vessels/topcoat | silicone | acrylic | incompatible | incompatible | verdict_agree |
| vessels/topcoat | silicone | vinyl | incompatible | incompatible | verdict_agree |
| vessels/topcoat | acrylic | epoxy | incompatible | incompatible | verdict_agree |
| vessels/topcoat | acrylic | polyurethane | incompatible | incompatible | verdict_agree |
| vessels/topcoat | acrylic | silicone | incompatible | incompatible | verdict_agree |
| vessels/topcoat | acrylic | vinyl | incompatible | incompatible | verdict_agree |
| vessels/topcoat | vinyl | epoxy | incompatible | incompatible | verdict_agree |
| vessels/topcoat | vinyl | polyurethane | incompatible | incompatible | verdict_agree |
| vessels/topcoat | vinyl | silicone | incompatible | incompatible | verdict_agree |
| vessels/topcoat | vinyl | acrylic | incompatible | incompatible | verdict_agree |
| vessels/registry port | Rotterdam | Hamburg | incompatible | incompatible | verdict_agree |
| vessels/registry port | Rotterdam | Antwerp | incompatible | incompatible | verdict_agree |
| vessels/registry port | Rotterdam | Bergen | incompatible | incompatible | verdict_agree |
| vessels/registry port | Rotterdam | Gothenburg | incompatible | incompatible | verdict_agree |
| vessels/registry port | Hamburg | Rotterdam | incompatible | incompatible | verdict_agree |
| vessels/registry port | Hamburg | Antwerp | incompatible | incompatible | verdict_agree |
| vessels/registry port | Hamburg | Bergen | incompatible | incompatible | verdict_agree |
| vessels/registry port | Hamburg | Gothenburg | incompatible | incompatible | verdict_agree |
| vessels/registry port | Antwerp | Rotterdam | incompatible | incompatible | verdict_agree |
| vessels/registry port | Antwerp | Hamburg | incompatible | incompatible | verdict_agree |
| vessels/registry port | Antwerp | Bergen | incompatible | incompatible | verdict_agree |
| vessels/registry port | Antwerp | Gothenburg | incompatible | incompatible | verdict_agree |
| vessels/registry port | Bergen | Rotterdam | incompatible | incompatible | verdict_agree |
| vessels/registry port | Bergen | Hamburg | incompatible | incompatible | verdict_agree |
| vessels/registry port | Bergen | Antwerp | incompatible | incompatible | verdict_agree |
| vessels/registry port | Bergen | Gothenburg | incompatible | incompatible | verdict_agree |
| vessels/registry port | Gothenburg | Rotterdam | incompatible | incompatible | verdict_agree |
| vessels/registry port | Gothenburg | Hamburg | incompatible | incompatible | verdict_agree |
| vessels/registry port | Gothenburg | Antwerp | incompatible | incompatible | verdict_agree |
| vessels/registry port | Gothenburg | Bergen | incompatible | incompatible | verdict_agree |
| vessels/ice class | pc1 | pc3 | incompatible | incompatible | verdict_agree |
| vessels/ice class | pc1 | pc5 | incompatible | incompatible | verdict_agree |
| vessels/ice class | pc1 | pc7 | incompatible | incompatible | verdict_agree |
| vessels/ice class | pc1 | noclass | incompatible | incompatible | verdict_agree |
| vessels/ice class | pc3 | pc1 | incompatible | incompatible | verdict_agree |
| vessels/ice class | pc3 | pc5 | incompatible | incompatible | verdict_agree |
| vessels/ice class | pc3 | pc7 | incompatible | incompatible | verdict_agree |
| vessels/ice class | pc3 | noclass | incompatible | incompatible | verdict_agree |
| vessels/ice class | pc5 | pc1 | incompatible | incompatible | verdict_agree |
| vessels/ice class | pc5 | pc3 | incompatible | incompatible | verdict_agree |
| vessels/ice class | pc5 | pc7 | incompatible | incompatible | verdict_agree |
| vessels/ice class | pc5 | noclass | incompatible | incompatible | verdict_agree |
| vessels/ice class | pc7 | pc1 | incompatible | incompatible | verdict_agree |
| vessels/ice class | pc7 | pc3 | incompatible | incompatible | verdict_agree |
| vessels/ice class | pc7 | pc5 | incompatible | incompatible | verdict_agree |
| vessels/ice class | pc7 | noclass | incompatible | incompatible | verdict_agree |
| vessels/ice class | noclass | pc1 | incompatible | incompatible | verdict_agree |
| vessels/ice class | noclass | pc3 | incompatible | incompatible | verdict_agree |
| vessels/ice class | noclass | pc5 | incompatible | incompatible | verdict_agree |
| vessels/ice class | noclass | pc7 | incompatible | incompatible | verdict_agree |
| vessels/propulsion type | diesel | steam | incompatible | incompatible | verdict_agree |
| vessels/propulsion type | diesel | sail | incompatible | incompatible | verdict_agree |
| vessels/propulsion type | diesel | gasturbine | incompatible | incompatible | verdict_agree |
| vessels/propulsion type | diesel | battery | incompatible | incompatible | verdict_agree |
| vessels/propulsion type | steam | diesel | incompatible | incompatible | verdict_agree |
| vessels/propulsion type | steam | sail | incompatible | incompatible | verdict_agree |
| vessels/propulsion type | steam | gasturbine | incompatible | incompatible | verdict_agree |
| vessels/propulsion type | steam | battery | incompatible | incompatible | verdict_agree |
| vessels/propulsion type | sail | diesel | incompatible | incompatible | verdict_agree |
| vessels/propulsion type | sail | steam | incompatible | incompatible | verdict_agree |
| vessels/propulsion type | sail | gasturbine | incompatible | incompatible | verdict_agree |
| vessels/propulsion type | sail | battery | incompatible | incompatible | verdict_agree |
| vessels/propulsion type | gasturbine | diesel | incompatible | incompatible | verdict_agree |
| vessels/propulsion type | gasturbine | steam | incompatible | incompatible | verdict_agree |
| vessels/propulsion type | gasturbine | sail | incompatible | incompatible | verdict_agree |
| vessels/propulsion type | gasturbine | battery | incompatible | incompatible | verdict_agree |
| vessels/propulsion type | battery | diesel | incompatible | incompatible | verdict_agree |
| vessels/propulsion type | battery | steam | incompatible | incompatible | verdict_agree |
| vessels/propulsion type | battery | sail | incompatible | incompatible | verdict_agree |
| vessels/propulsion type | battery | gasturbine | incompatible | incompatible | verdict_agree |
| furnaces/damper position | closed | twentyfive | incompatible | incompatible | verdict_agree |
| furnaces/damper position | closed | fifty | incompatible | incompatible | verdict_agree |
| furnaces/damper position | closed | seventyfive | incompatible | incompatible | verdict_agree |
| furnaces/damper position | closed | open | incompatible | incompatible | verdict_agree |
| furnaces/damper position | twentyfive | closed | length | unsure | length_stop_excluded_from_verdicts |
| furnaces/damper position | twentyfive | fifty | incompatible | incompatible | verdict_agree |
| furnaces/damper position | twentyfive | seventyfive | incompatible | incompatible | verdict_agree |
| furnaces/damper position | twentyfive | open | compatible | length | length_stop_excluded_from_verdicts |
| furnaces/damper position | fifty | closed | unsure | incompatible | verdict_disagree |
| furnaces/damper position | fifty | twentyfive | incompatible | incompatible | verdict_agree |
| furnaces/damper position | fifty | seventyfive | incompatible | incompatible | verdict_agree |
| furnaces/damper position | fifty | open | compatible | compatible | verdict_agree |
| furnaces/damper position | seventyfive | closed | length | incompatible | length_stop_excluded_from_verdicts |
| furnaces/damper position | seventyfive | twentyfive | incompatible | incompatible | verdict_agree |
| furnaces/damper position | seventyfive | fifty | incompatible | incompatible | verdict_agree |
| furnaces/damper position | seventyfive | open | compatible | compatible | verdict_agree |
| furnaces/damper position | open | closed | incompatible | incompatible | verdict_agree |
| furnaces/damper position | open | twentyfive | length | length | length_stop_excluded_from_verdicts |
| furnaces/damper position | open | fifty | length | unsure | length_stop_excluded_from_verdicts |
| furnaces/damper position | open | seventyfive | compatible | compatible | verdict_agree |
| furnaces/lining brick | alumina | magnesia | incompatible | incompatible | verdict_agree |
| furnaces/lining brick | alumina | zirconia | incompatible | incompatible | verdict_agree |
| furnaces/lining brick | alumina | silica | incompatible | incompatible | verdict_agree |
| furnaces/lining brick | alumina | chromite | incompatible | incompatible | verdict_agree |
| furnaces/lining brick | magnesia | alumina | incompatible | incompatible | verdict_agree |
| furnaces/lining brick | magnesia | zirconia | incompatible | incompatible | verdict_agree |
| furnaces/lining brick | magnesia | silica | incompatible | incompatible | verdict_agree |
| furnaces/lining brick | magnesia | chromite | incompatible | incompatible | verdict_agree |
| furnaces/lining brick | zirconia | alumina | incompatible | incompatible | verdict_agree |
| furnaces/lining brick | zirconia | magnesia | incompatible | incompatible | verdict_agree |
| furnaces/lining brick | zirconia | silica | incompatible | incompatible | verdict_agree |
| furnaces/lining brick | zirconia | chromite | incompatible | incompatible | verdict_agree |
| furnaces/lining brick | silica | alumina | incompatible | incompatible | verdict_agree |
| furnaces/lining brick | silica | magnesia | incompatible | incompatible | verdict_agree |
| furnaces/lining brick | silica | zirconia | incompatible | incompatible | verdict_agree |
| furnaces/lining brick | silica | chromite | incompatible | incompatible | verdict_agree |
| furnaces/lining brick | chromite | alumina | incompatible | incompatible | verdict_agree |
| furnaces/lining brick | chromite | magnesia | incompatible | incompatible | verdict_agree |
| furnaces/lining brick | chromite | zirconia | incompatible | incompatible | verdict_agree |
| furnaces/lining brick | chromite | silica | incompatible | incompatible | verdict_agree |
| furnaces/purge gas | argon | nitrogen | incompatible | incompatible | verdict_agree |
| furnaces/purge gas | argon | helium | incompatible | incompatible | verdict_agree |
| furnaces/purge gas | argon | hydrogen | incompatible | incompatible | verdict_agree |
| furnaces/purge gas | argon | carbondioxide | incompatible | incompatible | verdict_agree |
| furnaces/purge gas | nitrogen | argon | incompatible | incompatible | verdict_agree |
| furnaces/purge gas | nitrogen | helium | incompatible | incompatible | verdict_agree |
| furnaces/purge gas | nitrogen | hydrogen | incompatible | incompatible | verdict_agree |
| furnaces/purge gas | nitrogen | carbondioxide | incompatible | incompatible | verdict_agree |
| furnaces/purge gas | helium | argon | incompatible | incompatible | verdict_agree |
| furnaces/purge gas | helium | nitrogen | incompatible | incompatible | verdict_agree |
| furnaces/purge gas | helium | hydrogen | incompatible | incompatible | verdict_agree |
| furnaces/purge gas | helium | carbondioxide | incompatible | incompatible | verdict_agree |
| furnaces/purge gas | hydrogen | argon | incompatible | incompatible | verdict_agree |
| furnaces/purge gas | hydrogen | nitrogen | incompatible | incompatible | verdict_agree |
| furnaces/purge gas | hydrogen | helium | incompatible | incompatible | verdict_agree |
| furnaces/purge gas | hydrogen | carbondioxide | incompatible | incompatible | verdict_agree |
| furnaces/purge gas | carbondioxide | argon | incompatible | incompatible | verdict_agree |
| furnaces/purge gas | carbondioxide | nitrogen | incompatible | incompatible | verdict_agree |
| furnaces/purge gas | carbondioxide | helium | incompatible | incompatible | verdict_agree |
| furnaces/purge gas | carbondioxide | hydrogen | incompatible | incompatible | verdict_agree |
| furnaces/thermocouple type | typej | typek | incompatible | incompatible | verdict_agree |
| furnaces/thermocouple type | typej | typen | incompatible | incompatible | verdict_agree |
| furnaces/thermocouple type | typej | typer | incompatible | incompatible | verdict_agree |
| furnaces/thermocouple type | typej | types | incompatible | incompatible | verdict_agree |
| furnaces/thermocouple type | typek | typej | incompatible | incompatible | verdict_agree |
| furnaces/thermocouple type | typek | typen | incompatible | incompatible | verdict_agree |
| furnaces/thermocouple type | typek | typer | incompatible | incompatible | verdict_agree |
| furnaces/thermocouple type | typek | types | incompatible | incompatible | verdict_agree |
| furnaces/thermocouple type | typen | typej | incompatible | incompatible | verdict_agree |
| furnaces/thermocouple type | typen | typek | incompatible | incompatible | verdict_agree |
| furnaces/thermocouple type | typen | typer | incompatible | incompatible | verdict_agree |
| furnaces/thermocouple type | typen | types | incompatible | incompatible | verdict_agree |
| furnaces/thermocouple type | typer | typej | incompatible | incompatible | verdict_agree |
| furnaces/thermocouple type | typer | typek | incompatible | incompatible | verdict_agree |
| furnaces/thermocouple type | typer | typen | incompatible | incompatible | verdict_agree |
| furnaces/thermocouple type | typer | types | incompatible | incompatible | verdict_agree |
| furnaces/thermocouple type | types | typej | incompatible | incompatible | verdict_agree |
| furnaces/thermocouple type | types | typek | incompatible | incompatible | verdict_agree |
| furnaces/thermocouple type | types | typen | incompatible | incompatible | verdict_agree |
| furnaces/thermocouple type | types | typer | length | incompatible | length_stop_excluded_from_verdicts |
| furnaces/hearth drive | roller | walking | incompatible | incompatible | verdict_agree |
| furnaces/hearth drive | roller | pusher | incompatible | incompatible | verdict_agree |
| furnaces/hearth drive | roller | rotary | incompatible | compatible | verdict_disagree |
| furnaces/hearth drive | roller | car | incompatible | incompatible | verdict_agree |
| furnaces/hearth drive | walking | roller | incompatible | incompatible | verdict_agree |
| furnaces/hearth drive | walking | pusher | unsure | incompatible | verdict_disagree |
| furnaces/hearth drive | walking | rotary | incompatible | incompatible | verdict_agree |
| furnaces/hearth drive | walking | car | incompatible | incompatible | verdict_agree |
| furnaces/hearth drive | pusher | roller | incompatible | incompatible | verdict_agree |
| furnaces/hearth drive | pusher | walking | incompatible | incompatible | verdict_agree |
| furnaces/hearth drive | pusher | rotary | incompatible | incompatible | verdict_agree |
| furnaces/hearth drive | pusher | car | incompatible | incompatible | verdict_agree |
| furnaces/hearth drive | rotary | roller | incompatible | length | length_stop_excluded_from_verdicts |
| furnaces/hearth drive | rotary | walking | incompatible | incompatible | verdict_agree |
| furnaces/hearth drive | rotary | pusher | incompatible | incompatible | verdict_agree |
| furnaces/hearth drive | rotary | car | incompatible | incompatible | verdict_agree |
| furnaces/hearth drive | car | roller | incompatible | incompatible | verdict_agree |
| furnaces/hearth drive | car | walking | incompatible | incompatible | verdict_agree |
| furnaces/hearth drive | car | pusher | incompatible | incompatible | verdict_agree |
| furnaces/hearth drive | car | rotary | incompatible | incompatible | verdict_agree |
| reactors/loop state | drained | cold | compatible | compatible | verdict_agree |
| reactors/loop state | drained | heatup | incompatible | incompatible | verdict_agree |
| reactors/loop state | drained | hot | unsure | incompatible | verdict_disagree |
| reactors/loop state | drained | cooldown | incompatible | unsure | verdict_disagree |
| reactors/loop state | cold | drained | compatible | compatible | verdict_agree |
| reactors/loop state | cold | heatup | incompatible | incompatible | verdict_agree |
| reactors/loop state | cold | hot | incompatible | incompatible | verdict_agree |
| reactors/loop state | cold | cooldown | incompatible | length | length_stop_excluded_from_verdicts |
| reactors/loop state | heatup | drained | incompatible | incompatible | verdict_agree |
| reactors/loop state | heatup | cold | incompatible | incompatible | verdict_agree |
| reactors/loop state | heatup | hot | compatible | length | length_stop_excluded_from_verdicts |
| reactors/loop state | heatup | cooldown | incompatible | incompatible | verdict_agree |
| reactors/loop state | hot | drained | incompatible | unsure | verdict_disagree |
| reactors/loop state | hot | cold | incompatible | incompatible | verdict_agree |
| reactors/loop state | hot | heatup | compatible | compatible | verdict_agree |
| reactors/loop state | hot | cooldown | incompatible | incompatible | verdict_agree |
| reactors/loop state | cooldown | drained | incompatible | incompatible | verdict_agree |
| reactors/loop state | cooldown | cold | length | unsure | length_stop_excluded_from_verdicts |
| reactors/loop state | cooldown | heatup | incompatible | incompatible | verdict_agree |
| reactors/loop state | cooldown | hot | incompatible | incompatible | verdict_agree |
| reactors/cladding metal | zircaloy | ss316 | incompatible | incompatible | verdict_agree |
| reactors/cladding metal | zircaloy | magnox | incompatible | incompatible | verdict_agree |
| reactors/cladding metal | zircaloy | inconel | incompatible | incompatible | verdict_agree |
| reactors/cladding metal | zircaloy | aluminum | incompatible | incompatible | verdict_agree |
| reactors/cladding metal | ss316 | zircaloy | incompatible | incompatible | verdict_agree |
| reactors/cladding metal | ss316 | magnox | incompatible | incompatible | verdict_agree |
| reactors/cladding metal | ss316 | inconel | incompatible | incompatible | verdict_agree |
| reactors/cladding metal | ss316 | aluminum | incompatible | incompatible | verdict_agree |
| reactors/cladding metal | magnox | zircaloy | incompatible | incompatible | verdict_agree |
| reactors/cladding metal | magnox | ss316 | incompatible | incompatible | verdict_agree |
| reactors/cladding metal | magnox | inconel | incompatible | incompatible | verdict_agree |
| reactors/cladding metal | magnox | aluminum | incompatible | incompatible | verdict_agree |
| reactors/cladding metal | inconel | zircaloy | incompatible | incompatible | verdict_agree |
| reactors/cladding metal | inconel | ss316 | incompatible | incompatible | verdict_agree |
| reactors/cladding metal | inconel | magnox | incompatible | incompatible | verdict_agree |
| reactors/cladding metal | inconel | aluminum | incompatible | incompatible | verdict_agree |
| reactors/cladding metal | aluminum | zircaloy | incompatible | incompatible | verdict_agree |
| reactors/cladding metal | aluminum | ss316 | incompatible | incompatible | verdict_agree |
| reactors/cladding metal | aluminum | magnox | incompatible | incompatible | verdict_agree |
| reactors/cladding metal | aluminum | inconel | incompatible | incompatible | verdict_agree |
| reactors/moderator type | graphite | heavywater | incompatible | incompatible | verdict_agree |
| reactors/moderator type | graphite | lightwater | incompatible | incompatible | verdict_agree |
| reactors/moderator type | graphite | beryllium | incompatible | incompatible | verdict_agree |
| reactors/moderator type | graphite | paraffin | incompatible | incompatible | verdict_agree |
| reactors/moderator type | heavywater | graphite | incompatible | incompatible | verdict_agree |
| reactors/moderator type | heavywater | lightwater | incompatible | incompatible | verdict_agree |
| reactors/moderator type | heavywater | beryllium | incompatible | incompatible | verdict_agree |
| reactors/moderator type | heavywater | paraffin | incompatible | incompatible | verdict_agree |
| reactors/moderator type | lightwater | graphite | incompatible | incompatible | verdict_agree |
| reactors/moderator type | lightwater | heavywater | incompatible | incompatible | verdict_agree |
| reactors/moderator type | lightwater | beryllium | incompatible | incompatible | verdict_agree |
| reactors/moderator type | lightwater | paraffin | incompatible | incompatible | verdict_agree |
| reactors/moderator type | beryllium | graphite | incompatible | incompatible | verdict_agree |
| reactors/moderator type | beryllium | heavywater | incompatible | incompatible | verdict_agree |
| reactors/moderator type | beryllium | lightwater | incompatible | incompatible | verdict_agree |
| reactors/moderator type | beryllium | paraffin | incompatible | incompatible | verdict_agree |
| reactors/moderator type | paraffin | graphite | incompatible | incompatible | verdict_agree |
| reactors/moderator type | paraffin | heavywater | incompatible | incompatible | verdict_agree |
| reactors/moderator type | paraffin | lightwater | incompatible | incompatible | verdict_agree |
| reactors/moderator type | paraffin | beryllium | incompatible | incompatible | verdict_agree |
| reactors/shielding material | concrete | lead | incompatible | incompatible | verdict_agree |
| reactors/shielding material | concrete | tungsten | incompatible | incompatible | verdict_agree |
| reactors/shielding material | concrete | steel | incompatible | incompatible | verdict_agree |
| reactors/shielding material | concrete | depleteduranium | incompatible | incompatible | verdict_agree |
| reactors/shielding material | lead | concrete | incompatible | incompatible | verdict_agree |
| reactors/shielding material | lead | tungsten | incompatible | incompatible | verdict_agree |
| reactors/shielding material | lead | steel | incompatible | incompatible | verdict_agree |
| reactors/shielding material | lead | depleteduranium | incompatible | incompatible | verdict_agree |
| reactors/shielding material | tungsten | concrete | incompatible | incompatible | verdict_agree |
| reactors/shielding material | tungsten | lead | incompatible | incompatible | verdict_agree |
| reactors/shielding material | tungsten | steel | incompatible | incompatible | verdict_agree |
| reactors/shielding material | tungsten | depleteduranium | incompatible | incompatible | verdict_agree |
| reactors/shielding material | steel | concrete | incompatible | incompatible | verdict_agree |
| reactors/shielding material | steel | lead | incompatible | incompatible | verdict_agree |
| reactors/shielding material | steel | tungsten | incompatible | incompatible | verdict_agree |
| reactors/shielding material | steel | depleteduranium | incompatible | incompatible | verdict_agree |
| reactors/shielding material | depleteduranium | concrete | incompatible | incompatible | verdict_agree |
| reactors/shielding material | depleteduranium | lead | incompatible | incompatible | verdict_agree |
| reactors/shielding material | depleteduranium | tungsten | incompatible | incompatible | verdict_agree |
| reactors/shielding material | depleteduranium | steel | incompatible | incompatible | verdict_agree |
| reactors/coolant type | water | sodium | incompatible | incompatible | verdict_agree |
| reactors/coolant type | water | helium | incompatible | incompatible | verdict_agree |
| reactors/coolant type | water | carbondioxide | incompatible | incompatible | verdict_agree |
| reactors/coolant type | water | leadbismuth | incompatible | incompatible | verdict_agree |
| reactors/coolant type | sodium | water | incompatible | incompatible | verdict_agree |
| reactors/coolant type | sodium | helium | incompatible | incompatible | verdict_agree |
| reactors/coolant type | sodium | carbondioxide | incompatible | incompatible | verdict_agree |
| reactors/coolant type | sodium | leadbismuth | incompatible | incompatible | verdict_agree |
| reactors/coolant type | helium | water | incompatible | incompatible | verdict_agree |
| reactors/coolant type | helium | sodium | incompatible | incompatible | verdict_agree |
| reactors/coolant type | helium | carbondioxide | incompatible | incompatible | verdict_agree |
| reactors/coolant type | helium | leadbismuth | incompatible | incompatible | verdict_agree |
| reactors/coolant type | carbondioxide | water | incompatible | incompatible | verdict_agree |
| reactors/coolant type | carbondioxide | sodium | incompatible | incompatible | verdict_agree |
| reactors/coolant type | carbondioxide | helium | incompatible | incompatible | verdict_agree |
| reactors/coolant type | carbondioxide | leadbismuth | incompatible | incompatible | verdict_agree |
| reactors/coolant type | leadbismuth | water | incompatible | incompatible | verdict_agree |
| reactors/coolant type | leadbismuth | sodium | incompatible | incompatible | verdict_agree |
| reactors/coolant type | leadbismuth | helium | incompatible | incompatible | verdict_agree |
| reactors/coolant type | leadbismuth | carbondioxide | incompatible | incompatible | verdict_agree |
| compressors/compressor mode | stopped | starting | incompatible | incompatible | verdict_agree |
| compressors/compressor mode | stopped | onload | incompatible | incompatible | verdict_agree |
| compressors/compressor mode | stopped | offload | incompatible | incompatible | verdict_agree |
| compressors/compressor mode | stopped | stopping | incompatible | incompatible | verdict_agree |
| compressors/compressor mode | starting | stopped | incompatible | incompatible | verdict_agree |
| compressors/compressor mode | starting | onload | incompatible | incompatible | verdict_agree |
| compressors/compressor mode | starting | offload | incompatible | compatible | verdict_disagree |
| compressors/compressor mode | starting | stopping | incompatible | incompatible | verdict_agree |
| compressors/compressor mode | onload | stopped | incompatible | incompatible | verdict_agree |
| compressors/compressor mode | onload | starting | incompatible | incompatible | verdict_agree |
| compressors/compressor mode | onload | offload | incompatible | incompatible | verdict_agree |
| compressors/compressor mode | onload | stopping | incompatible | incompatible | verdict_agree |
| compressors/compressor mode | offload | stopped | incompatible | incompatible | verdict_agree |
| compressors/compressor mode | offload | starting | incompatible | incompatible | verdict_agree |
| compressors/compressor mode | offload | onload | incompatible | incompatible | verdict_agree |
| compressors/compressor mode | offload | stopping | incompatible | length | length_stop_excluded_from_verdicts |
| compressors/compressor mode | stopping | stopped | incompatible | incompatible | verdict_agree |
| compressors/compressor mode | stopping | starting | incompatible | incompatible | verdict_agree |
| compressors/compressor mode | stopping | onload | incompatible | incompatible | verdict_agree |
| compressors/compressor mode | stopping | offload | incompatible | incompatible | verdict_agree |
| compressors/rotor coating | ptfe | epoxyphenolic | incompatible | incompatible | verdict_agree |
| compressors/rotor coating | ptfe | aluminide | incompatible | incompatible | verdict_agree |
| compressors/rotor coating | ptfe | diamondlike | incompatible | incompatible | verdict_agree |
| compressors/rotor coating | ptfe | anodized | incompatible | incompatible | verdict_agree |
| compressors/rotor coating | epoxyphenolic | ptfe | incompatible | incompatible | verdict_agree |
| compressors/rotor coating | epoxyphenolic | aluminide | incompatible | incompatible | verdict_agree |
| compressors/rotor coating | epoxyphenolic | diamondlike | incompatible | incompatible | verdict_agree |
| compressors/rotor coating | epoxyphenolic | anodized | incompatible | incompatible | verdict_agree |
| compressors/rotor coating | aluminide | ptfe | incompatible | incompatible | verdict_agree |
| compressors/rotor coating | aluminide | epoxyphenolic | incompatible | incompatible | verdict_agree |
| compressors/rotor coating | aluminide | diamondlike | incompatible | incompatible | verdict_agree |
| compressors/rotor coating | aluminide | anodized | incompatible | incompatible | verdict_agree |
| compressors/rotor coating | diamondlike | ptfe | incompatible | incompatible | verdict_agree |
| compressors/rotor coating | diamondlike | epoxyphenolic | compatible | incompatible | verdict_disagree |
| compressors/rotor coating | diamondlike | aluminide | incompatible | incompatible | verdict_agree |
| compressors/rotor coating | diamondlike | anodized | length | incompatible | length_stop_excluded_from_verdicts |
| compressors/rotor coating | anodized | ptfe | compatible | incompatible | verdict_disagree |
| compressors/rotor coating | anodized | epoxyphenolic | incompatible | incompatible | verdict_agree |
| compressors/rotor coating | anodized | aluminide | incompatible | incompatible | verdict_agree |
| compressors/rotor coating | anodized | diamondlike | incompatible | incompatible | verdict_agree |
| compressors/intercooler medium | air | water | incompatible | incompatible | verdict_agree |
| compressors/intercooler medium | air | oil | incompatible | incompatible | verdict_agree |
| compressors/intercooler medium | air | ammonia | incompatible | incompatible | verdict_agree |
| compressors/intercooler medium | air | propane | incompatible | incompatible | verdict_agree |
| compressors/intercooler medium | water | air | incompatible | incompatible | verdict_agree |
| compressors/intercooler medium | water | oil | incompatible | incompatible | verdict_agree |
| compressors/intercooler medium | water | ammonia | incompatible | incompatible | verdict_agree |
| compressors/intercooler medium | water | propane | incompatible | incompatible | verdict_agree |
| compressors/intercooler medium | oil | air | incompatible | incompatible | verdict_agree |
| compressors/intercooler medium | oil | water | incompatible | incompatible | verdict_agree |
| compressors/intercooler medium | oil | ammonia | incompatible | incompatible | verdict_agree |
| compressors/intercooler medium | oil | propane | incompatible | incompatible | verdict_agree |
| compressors/intercooler medium | ammonia | air | incompatible | incompatible | verdict_agree |
| compressors/intercooler medium | ammonia | water | incompatible | incompatible | verdict_agree |
| compressors/intercooler medium | ammonia | oil | incompatible | incompatible | verdict_agree |
| compressors/intercooler medium | ammonia | propane | incompatible | incompatible | verdict_agree |
| compressors/intercooler medium | propane | air | incompatible | incompatible | verdict_agree |
| compressors/intercooler medium | propane | water | incompatible | incompatible | verdict_agree |
| compressors/intercooler medium | propane | oil | incompatible | incompatible | verdict_agree |
| compressors/intercooler medium | propane | ammonia | incompatible | incompatible | verdict_agree |
| compressors/shaft seal | labyrinth | mechanical | incompatible | incompatible | verdict_agree |
| compressors/shaft seal | labyrinth | packed | incompatible | incompatible | verdict_agree |
| compressors/shaft seal | labyrinth | lip | incompatible | incompatible | verdict_agree |
| compressors/shaft seal | labyrinth | brush | incompatible | incompatible | verdict_agree |
| compressors/shaft seal | mechanical | labyrinth | incompatible | incompatible | verdict_agree |
| compressors/shaft seal | mechanical | packed | incompatible | incompatible | verdict_agree |
| compressors/shaft seal | mechanical | lip | incompatible | incompatible | verdict_agree |
| compressors/shaft seal | mechanical | brush | incompatible | incompatible | verdict_agree |
| compressors/shaft seal | packed | labyrinth | incompatible | incompatible | verdict_agree |
| compressors/shaft seal | packed | mechanical | incompatible | incompatible | verdict_agree |
| compressors/shaft seal | packed | lip | incompatible | incompatible | verdict_agree |
| compressors/shaft seal | packed | brush | incompatible | incompatible | verdict_agree |
| compressors/shaft seal | lip | labyrinth | incompatible | incompatible | verdict_agree |
| compressors/shaft seal | lip | mechanical | incompatible | incompatible | verdict_agree |
| compressors/shaft seal | lip | packed | incompatible | incompatible | verdict_agree |
| compressors/shaft seal | lip | brush | incompatible | incompatible | verdict_agree |
| compressors/shaft seal | brush | labyrinth | incompatible | incompatible | verdict_agree |
| compressors/shaft seal | brush | mechanical | incompatible | incompatible | verdict_agree |
| compressors/shaft seal | brush | packed | incompatible | incompatible | verdict_agree |
| compressors/shaft seal | brush | lip | incompatible | incompatible | verdict_agree |
| compressors/unloader mechanism | pneumatic | hydraulic | incompatible | incompatible | verdict_agree |
| compressors/unloader mechanism | pneumatic | electric | incompatible | incompatible | verdict_agree |
| compressors/unloader mechanism | pneumatic | spring | compatible | incompatible | verdict_disagree |
| compressors/unloader mechanism | pneumatic | manual | incompatible | incompatible | verdict_agree |
| compressors/unloader mechanism | hydraulic | pneumatic | incompatible | incompatible | verdict_agree |
| compressors/unloader mechanism | hydraulic | electric | incompatible | incompatible | verdict_agree |
| compressors/unloader mechanism | hydraulic | spring | length | incompatible | length_stop_excluded_from_verdicts |
| compressors/unloader mechanism | hydraulic | manual | incompatible | incompatible | verdict_agree |
| compressors/unloader mechanism | electric | pneumatic | incompatible | incompatible | verdict_agree |
| compressors/unloader mechanism | electric | hydraulic | incompatible | incompatible | verdict_agree |
| compressors/unloader mechanism | electric | spring | incompatible | compatible | verdict_disagree |
| compressors/unloader mechanism | electric | manual | incompatible | incompatible | verdict_agree |
| compressors/unloader mechanism | spring | pneumatic | incompatible | incompatible | verdict_agree |
| compressors/unloader mechanism | spring | hydraulic | incompatible | incompatible | verdict_agree |
| compressors/unloader mechanism | spring | electric | incompatible | incompatible | verdict_agree |
| compressors/unloader mechanism | spring | manual | compatible | compatible | verdict_agree |
| compressors/unloader mechanism | manual | pneumatic | incompatible | incompatible | verdict_agree |
| compressors/unloader mechanism | manual | hydraulic | incompatible | incompatible | verdict_agree |
| compressors/unloader mechanism | manual | electric | incompatible | incompatible | verdict_agree |
| compressors/unloader mechanism | manual | spring | length | compatible | length_stop_excluded_from_verdicts |
| transformers/tap position | lowest | lower | length | length | length_stop_excluded_from_verdicts |
| transformers/tap position | lowest | nominal | compatible | compatible | verdict_agree |
| transformers/tap position | lowest | higher | incompatible | incompatible | verdict_agree |
| transformers/tap position | lowest | highest | incompatible | incompatible | verdict_agree |
| transformers/tap position | lower | lowest | compatible | compatible | verdict_agree |
| transformers/tap position | lower | nominal | incompatible | compatible | verdict_disagree |
| transformers/tap position | lower | higher | incompatible | incompatible | verdict_agree |
| transformers/tap position | lower | highest | incompatible | incompatible | verdict_agree |
| transformers/tap position | nominal | lowest | compatible | compatible | verdict_agree |
| transformers/tap position | nominal | lower | incompatible | compatible | verdict_disagree |
| transformers/tap position | nominal | higher | incompatible | compatible | verdict_disagree |
| transformers/tap position | nominal | highest | length | incompatible | length_stop_excluded_from_verdicts |
| transformers/tap position | higher | lowest | incompatible | incompatible | verdict_agree |
| transformers/tap position | higher | lower | incompatible | incompatible | verdict_agree |
| transformers/tap position | higher | nominal | compatible | length | length_stop_excluded_from_verdicts |
| transformers/tap position | higher | highest | compatible | compatible | verdict_agree |
| transformers/tap position | highest | lowest | incompatible | incompatible | verdict_agree |
| transformers/tap position | highest | lower | incompatible | incompatible | verdict_agree |
| transformers/tap position | highest | nominal | length | compatible | length_stop_excluded_from_verdicts |
| transformers/tap position | highest | higher | compatible | compatible | verdict_agree |
| transformers/core steel | silicon | amorphous | length | compatible | length_stop_excluded_from_verdicts |
| transformers/core steel | silicon | ferrite | incompatible | incompatible | verdict_agree |
| transformers/core steel | silicon | permalloy | incompatible | incompatible | verdict_agree |
| transformers/core steel | silicon | nanocrystalline | compatible | compatible | verdict_agree |
| transformers/core steel | amorphous | silicon | compatible | incompatible | verdict_disagree |
| transformers/core steel | amorphous | ferrite | incompatible | incompatible | verdict_agree |
| transformers/core steel | amorphous | permalloy | incompatible | incompatible | verdict_agree |
| transformers/core steel | amorphous | nanocrystalline | incompatible | incompatible | verdict_agree |
| transformers/core steel | ferrite | silicon | incompatible | compatible | verdict_disagree |
| transformers/core steel | ferrite | amorphous | incompatible | incompatible | verdict_agree |
| transformers/core steel | ferrite | permalloy | incompatible | incompatible | verdict_agree |
| transformers/core steel | ferrite | nanocrystalline | incompatible | incompatible | verdict_agree |
| transformers/core steel | permalloy | silicon | incompatible | incompatible | verdict_agree |
| transformers/core steel | permalloy | amorphous | compatible | length | length_stop_excluded_from_verdicts |
| transformers/core steel | permalloy | ferrite | incompatible | incompatible | verdict_agree |
| transformers/core steel | permalloy | nanocrystalline | compatible | compatible | verdict_agree |
| transformers/core steel | nanocrystalline | silicon | compatible | compatible | verdict_agree |
| transformers/core steel | nanocrystalline | amorphous | incompatible | incompatible | verdict_agree |
| transformers/core steel | nanocrystalline | ferrite | incompatible | incompatible | verdict_agree |
| transformers/core steel | nanocrystalline | permalloy | compatible | length | length_stop_excluded_from_verdicts |
| transformers/dielectric fluid | fluorinated | silicone | compatible | incompatible | verdict_disagree |
| transformers/dielectric fluid | fluorinated | ester | compatible | compatible | verdict_agree |
| transformers/dielectric fluid | fluorinated | paraffinic | incompatible | incompatible | verdict_agree |
| transformers/dielectric fluid | fluorinated | naphthenic | incompatible | incompatible | verdict_agree |
| transformers/dielectric fluid | silicone | fluorinated | incompatible | length | length_stop_excluded_from_verdicts |
| transformers/dielectric fluid | silicone | ester | incompatible | incompatible | verdict_agree |
| transformers/dielectric fluid | silicone | paraffinic | incompatible | incompatible | verdict_agree |
| transformers/dielectric fluid | silicone | naphthenic | incompatible | incompatible | verdict_agree |
| transformers/dielectric fluid | ester | fluorinated | compatible | compatible | verdict_agree |
| transformers/dielectric fluid | ester | silicone | incompatible | incompatible | verdict_agree |
| transformers/dielectric fluid | ester | paraffinic | incompatible | incompatible | verdict_agree |
| transformers/dielectric fluid | ester | naphthenic | incompatible | length | length_stop_excluded_from_verdicts |
| transformers/dielectric fluid | paraffinic | fluorinated | incompatible | incompatible | verdict_agree |
| transformers/dielectric fluid | paraffinic | silicone | incompatible | incompatible | verdict_agree |
| transformers/dielectric fluid | paraffinic | ester | incompatible | incompatible | verdict_agree |
| transformers/dielectric fluid | paraffinic | naphthenic | incompatible | incompatible | verdict_agree |
| transformers/dielectric fluid | naphthenic | fluorinated | incompatible | incompatible | verdict_agree |
| transformers/dielectric fluid | naphthenic | silicone | incompatible | incompatible | verdict_agree |
| transformers/dielectric fluid | naphthenic | ester | incompatible | incompatible | verdict_agree |
| transformers/dielectric fluid | naphthenic | paraffinic | incompatible | incompatible | verdict_agree |
| transformers/bushing material | porcelain | oilpaper | incompatible | incompatible | verdict_agree |
| transformers/bushing material | porcelain | epoxy | incompatible | incompatible | verdict_agree |
| transformers/bushing material | porcelain | siliconerubber | incompatible | incompatible | verdict_agree |
| transformers/bushing material | porcelain | glass | incompatible | incompatible | verdict_agree |
| transformers/bushing material | oilpaper | porcelain | incompatible | incompatible | verdict_agree |
| transformers/bushing material | oilpaper | epoxy | incompatible | incompatible | verdict_agree |
| transformers/bushing material | oilpaper | siliconerubber | incompatible | incompatible | verdict_agree |
| transformers/bushing material | oilpaper | glass | incompatible | incompatible | verdict_agree |
| transformers/bushing material | epoxy | porcelain | incompatible | incompatible | verdict_agree |
| transformers/bushing material | epoxy | oilpaper | incompatible | incompatible | verdict_agree |
| transformers/bushing material | epoxy | siliconerubber | incompatible | incompatible | verdict_agree |
| transformers/bushing material | epoxy | glass | incompatible | incompatible | verdict_agree |
| transformers/bushing material | siliconerubber | porcelain | incompatible | incompatible | verdict_agree |
| transformers/bushing material | siliconerubber | oilpaper | incompatible | incompatible | verdict_agree |
| transformers/bushing material | siliconerubber | epoxy | incompatible | incompatible | verdict_agree |
| transformers/bushing material | siliconerubber | glass | incompatible | incompatible | verdict_agree |
| transformers/bushing material | glass | porcelain | incompatible | incompatible | verdict_agree |
| transformers/bushing material | glass | oilpaper | incompatible | incompatible | verdict_agree |
| transformers/bushing material | glass | epoxy | incompatible | incompatible | verdict_agree |
| transformers/bushing material | glass | siliconerubber | incompatible | incompatible | verdict_agree |
| transformers/vector group | dyn11 | yyn0 | incompatible | incompatible | verdict_agree |
| transformers/vector group | dyn11 | ynd1 | incompatible | incompatible | verdict_agree |
| transformers/vector group | dyn11 | dd0 | incompatible | incompatible | verdict_agree |
| transformers/vector group | dyn11 | yzn11 | incompatible | incompatible | verdict_agree |
| transformers/vector group | yyn0 | dyn11 | incompatible | incompatible | verdict_agree |
| transformers/vector group | yyn0 | ynd1 | incompatible | incompatible | verdict_agree |
| transformers/vector group | yyn0 | dd0 | incompatible | incompatible | verdict_agree |
| transformers/vector group | yyn0 | yzn11 | incompatible | incompatible | verdict_agree |
| transformers/vector group | ynd1 | dyn11 | incompatible | incompatible | verdict_agree |
| transformers/vector group | ynd1 | yyn0 | incompatible | incompatible | verdict_agree |
| transformers/vector group | ynd1 | dd0 | incompatible | incompatible | verdict_agree |
| transformers/vector group | ynd1 | yzn11 | incompatible | incompatible | verdict_agree |
| transformers/vector group | dd0 | dyn11 | incompatible | incompatible | verdict_agree |
| transformers/vector group | dd0 | yyn0 | incompatible | incompatible | verdict_agree |
| transformers/vector group | dd0 | ynd1 | incompatible | incompatible | verdict_agree |
| transformers/vector group | dd0 | yzn11 | incompatible | incompatible | verdict_agree |
| transformers/vector group | yzn11 | dyn11 | incompatible | incompatible | verdict_agree |
| transformers/vector group | yzn11 | yyn0 | incompatible | incompatible | verdict_agree |
| transformers/vector group | yzn11 | ynd1 | incompatible | incompatible | verdict_agree |
| transformers/vector group | yzn11 | dd0 | incompatible | incompatible | verdict_agree |
| boilers/burner state | prepurge | lightoff | incompatible | incompatible | verdict_agree |
| boilers/burner state | prepurge | firing | incompatible | incompatible | verdict_agree |
| boilers/burner state | prepurge | postpurge | incompatible | incompatible | verdict_agree |
| boilers/burner state | prepurge | lockout | incompatible | incompatible | verdict_agree |
| boilers/burner state | lightoff | prepurge | incompatible | incompatible | verdict_agree |
| boilers/burner state | lightoff | firing | incompatible | incompatible | verdict_agree |
| boilers/burner state | lightoff | postpurge | incompatible | incompatible | verdict_agree |
| boilers/burner state | lightoff | lockout | incompatible | incompatible | verdict_agree |
| boilers/burner state | firing | prepurge | incompatible | incompatible | verdict_agree |
| boilers/burner state | firing | lightoff | incompatible | incompatible | verdict_agree |
| boilers/burner state | firing | postpurge | incompatible | incompatible | verdict_agree |
| boilers/burner state | firing | lockout | incompatible | incompatible | verdict_agree |
| boilers/burner state | postpurge | prepurge | incompatible | incompatible | verdict_agree |
| boilers/burner state | postpurge | lightoff | incompatible | incompatible | verdict_agree |
| boilers/burner state | postpurge | firing | incompatible | incompatible | verdict_agree |
| boilers/burner state | postpurge | lockout | incompatible | incompatible | verdict_agree |
| boilers/burner state | lockout | prepurge | incompatible | incompatible | verdict_agree |
| boilers/burner state | lockout | lightoff | incompatible | incompatible | verdict_agree |
| boilers/burner state | lockout | firing | incompatible | incompatible | verdict_agree |
| boilers/burner state | lockout | postpurge | incompatible | incompatible | verdict_agree |
| boilers/tube alloy | t11 | t22 | incompatible | incompatible | verdict_agree |
| boilers/tube alloy | t11 | t91 | incompatible | incompatible | verdict_agree |
| boilers/tube alloy | t11 | tp304h | incompatible | incompatible | verdict_agree |
| boilers/tube alloy | t11 | inconel625 | incompatible | incompatible | verdict_agree |
| boilers/tube alloy | t22 | t11 | incompatible | incompatible | verdict_agree |
| boilers/tube alloy | t22 | t91 | incompatible | incompatible | verdict_agree |
| boilers/tube alloy | t22 | tp304h | incompatible | incompatible | verdict_agree |
| boilers/tube alloy | t22 | inconel625 | incompatible | incompatible | verdict_agree |
| boilers/tube alloy | t91 | t11 | incompatible | incompatible | verdict_agree |
| boilers/tube alloy | t91 | t22 | incompatible | incompatible | verdict_agree |
| boilers/tube alloy | t91 | tp304h | incompatible | incompatible | verdict_agree |
| boilers/tube alloy | t91 | inconel625 | incompatible | incompatible | verdict_agree |
| boilers/tube alloy | tp304h | t11 | incompatible | incompatible | verdict_agree |
| boilers/tube alloy | tp304h | t22 | incompatible | incompatible | verdict_agree |
| boilers/tube alloy | tp304h | t91 | incompatible | incompatible | verdict_agree |
| boilers/tube alloy | tp304h | inconel625 | incompatible | incompatible | verdict_agree |
| boilers/tube alloy | inconel625 | t11 | incompatible | incompatible | verdict_agree |
| boilers/tube alloy | inconel625 | t22 | incompatible | incompatible | verdict_agree |
| boilers/tube alloy | inconel625 | t91 | incompatible | incompatible | verdict_agree |
| boilers/tube alloy | inconel625 | tp304h | incompatible | incompatible | verdict_agree |
| boilers/feedwater program | phosphate | chelant | incompatible | incompatible | verdict_agree |
| boilers/feedwater program | phosphate | volatile | incompatible | incompatible | verdict_agree |
| boilers/feedwater program | phosphate | oxygenated | incompatible | incompatible | verdict_agree |
| boilers/feedwater program | phosphate | caustic | incompatible | incompatible | verdict_agree |
| boilers/feedwater program | chelant | phosphate | incompatible | incompatible | verdict_agree |
| boilers/feedwater program | chelant | volatile | incompatible | incompatible | verdict_agree |
| boilers/feedwater program | chelant | oxygenated | compatible | compatible | verdict_agree |
| boilers/feedwater program | chelant | caustic | compatible | compatible | verdict_agree |
| boilers/feedwater program | volatile | phosphate | compatible | incompatible | verdict_disagree |
| boilers/feedwater program | volatile | chelant | length | compatible | length_stop_excluded_from_verdicts |
| boilers/feedwater program | volatile | oxygenated | compatible | compatible | verdict_agree |
| boilers/feedwater program | volatile | caustic | compatible | compatible | verdict_agree |
| boilers/feedwater program | oxygenated | phosphate | compatible | compatible | verdict_agree |
| boilers/feedwater program | oxygenated | chelant | compatible | compatible | verdict_agree |
| boilers/feedwater program | oxygenated | volatile | compatible | compatible | verdict_agree |
| boilers/feedwater program | oxygenated | caustic | compatible | compatible | verdict_agree |
| boilers/feedwater program | caustic | phosphate | compatible | compatible | verdict_agree |
| boilers/feedwater program | caustic | chelant | compatible | compatible | verdict_agree |
| boilers/feedwater program | caustic | volatile | compatible | compatible | verdict_agree |
| boilers/feedwater program | caustic | oxygenated | compatible | compatible | verdict_agree |
| boilers/sootblower type | retractable | rotary | compatible | compatible | verdict_agree |
| boilers/sootblower type | retractable | wall | incompatible | compatible | verdict_disagree |
| boilers/sootblower type | retractable | rake | incompatible | compatible | verdict_disagree |
| boilers/sootblower type | retractable | sonic | incompatible | compatible | verdict_disagree |
| boilers/sootblower type | rotary | retractable | compatible | compatible | verdict_agree |
| boilers/sootblower type | rotary | wall | incompatible | incompatible | verdict_agree |
| boilers/sootblower type | rotary | rake | incompatible | incompatible | verdict_agree |
| boilers/sootblower type | rotary | sonic | incompatible | incompatible | verdict_agree |
| boilers/sootblower type | wall | retractable | incompatible | compatible | verdict_disagree |
| boilers/sootblower type | wall | rotary | incompatible | incompatible | verdict_agree |
| boilers/sootblower type | wall | rake | incompatible | incompatible | verdict_agree |
| boilers/sootblower type | wall | sonic | length | incompatible | length_stop_excluded_from_verdicts |
| boilers/sootblower type | rake | retractable | length | compatible | length_stop_excluded_from_verdicts |
| boilers/sootblower type | rake | rotary | incompatible | incompatible | verdict_agree |
| boilers/sootblower type | rake | wall | incompatible | incompatible | verdict_agree |
| boilers/sootblower type | rake | sonic | incompatible | incompatible | verdict_agree |
| boilers/sootblower type | sonic | retractable | compatible | compatible | verdict_agree |
| boilers/sootblower type | sonic | rotary | incompatible | incompatible | verdict_agree |
| boilers/sootblower type | sonic | wall | incompatible | incompatible | verdict_agree |
| boilers/sootblower type | sonic | rake | incompatible | incompatible | verdict_agree |
| boilers/firing method | pulverized | stoker | incompatible | incompatible | verdict_agree |
| boilers/firing method | pulverized | fluidized | incompatible | incompatible | verdict_agree |
| boilers/firing method | pulverized | cyclone | incompatible | incompatible | verdict_agree |
| boilers/firing method | pulverized | gasfired | incompatible | incompatible | verdict_agree |
| boilers/firing method | stoker | pulverized | incompatible | incompatible | verdict_agree |
| boilers/firing method | stoker | fluidized | incompatible | incompatible | verdict_agree |
| boilers/firing method | stoker | cyclone | incompatible | incompatible | verdict_agree |
| boilers/firing method | stoker | gasfired | incompatible | incompatible | verdict_agree |
| boilers/firing method | fluidized | pulverized | incompatible | incompatible | verdict_agree |
| boilers/firing method | fluidized | stoker | incompatible | incompatible | verdict_agree |
| boilers/firing method | fluidized | cyclone | incompatible | incompatible | verdict_agree |
| boilers/firing method | fluidized | gasfired | compatible | compatible | verdict_agree |
| boilers/firing method | cyclone | pulverized | incompatible | incompatible | verdict_agree |
| boilers/firing method | cyclone | stoker | incompatible | incompatible | verdict_agree |
| boilers/firing method | cyclone | fluidized | incompatible | incompatible | verdict_agree |
| boilers/firing method | cyclone | gasfired | incompatible | incompatible | verdict_agree |
| boilers/firing method | gasfired | pulverized | incompatible | incompatible | verdict_agree |
| boilers/firing method | gasfired | stoker | incompatible | incompatible | verdict_agree |
| boilers/firing method | gasfired | fluidized | compatible | incompatible | verdict_disagree |
| boilers/firing method | gasfired | cyclone | incompatible | incompatible | verdict_agree |
| chillers/circuit state | idle | startup | incompatible | incompatible | verdict_agree |
| chillers/circuit state | idle | chilling | incompatible | incompatible | verdict_agree |
| chillers/circuit state | idle | defrosting | incompatible | incompatible | verdict_agree |
| chillers/circuit state | idle | pumpdown | incompatible | incompatible | verdict_agree |
| chillers/circuit state | startup | idle | incompatible | incompatible | verdict_agree |
| chillers/circuit state | startup | chilling | incompatible | incompatible | verdict_agree |
| chillers/circuit state | startup | defrosting | incompatible | incompatible | verdict_agree |
| chillers/circuit state | startup | pumpdown | incompatible | incompatible | verdict_agree |
| chillers/circuit state | chilling | idle | incompatible | incompatible | verdict_agree |
| chillers/circuit state | chilling | startup | incompatible | incompatible | verdict_agree |
| chillers/circuit state | chilling | defrosting | incompatible | incompatible | verdict_agree |
| chillers/circuit state | chilling | pumpdown | incompatible | incompatible | verdict_agree |
| chillers/circuit state | defrosting | idle | incompatible | incompatible | verdict_agree |
| chillers/circuit state | defrosting | startup | incompatible | incompatible | verdict_agree |
| chillers/circuit state | defrosting | chilling | incompatible | incompatible | verdict_agree |
| chillers/circuit state | defrosting | pumpdown | incompatible | incompatible | verdict_agree |
| chillers/circuit state | pumpdown | idle | incompatible | incompatible | verdict_agree |
| chillers/circuit state | pumpdown | startup | incompatible | incompatible | verdict_agree |
| chillers/circuit state | pumpdown | chilling | incompatible | incompatible | verdict_agree |
| chillers/circuit state | pumpdown | defrosting | incompatible | incompatible | verdict_agree |
| chillers/brazing filler | silver | nickel | incompatible | incompatible | verdict_agree |
| chillers/brazing filler | silver | aluminum | incompatible | incompatible | verdict_agree |
| chillers/brazing filler | silver | copper | incompatible | incompatible | verdict_agree |
| chillers/brazing filler | silver | gold | incompatible | incompatible | verdict_agree |
| chillers/brazing filler | nickel | silver | incompatible | incompatible | verdict_agree |
| chillers/brazing filler | nickel | aluminum | incompatible | incompatible | verdict_agree |
| chillers/brazing filler | nickel | copper | incompatible | incompatible | verdict_agree |
| chillers/brazing filler | nickel | gold | incompatible | incompatible | verdict_agree |
| chillers/brazing filler | aluminum | silver | incompatible | incompatible | verdict_agree |
| chillers/brazing filler | aluminum | nickel | incompatible | incompatible | verdict_agree |
| chillers/brazing filler | aluminum | copper | incompatible | incompatible | verdict_agree |
| chillers/brazing filler | aluminum | gold | incompatible | incompatible | verdict_agree |
| chillers/brazing filler | copper | silver | incompatible | incompatible | verdict_agree |
| chillers/brazing filler | copper | nickel | incompatible | incompatible | verdict_agree |
| chillers/brazing filler | copper | aluminum | incompatible | incompatible | verdict_agree |
| chillers/brazing filler | copper | gold | incompatible | incompatible | verdict_agree |
| chillers/brazing filler | gold | silver | incompatible | incompatible | verdict_agree |
| chillers/brazing filler | gold | nickel | incompatible | incompatible | verdict_agree |
| chillers/brazing filler | gold | aluminum | incompatible | incompatible | verdict_agree |
| chillers/brazing filler | gold | copper | incompatible | incompatible | verdict_agree |
| chillers/refrigerant | ammonia | propane | incompatible | incompatible | verdict_agree |
| chillers/refrigerant | ammonia | carbondioxide | incompatible | incompatible | verdict_agree |
| chillers/refrigerant | ammonia | isobutane | incompatible | incompatible | verdict_agree |
| chillers/refrigerant | ammonia | r134a | incompatible | incompatible | verdict_agree |
| chillers/refrigerant | propane | ammonia | incompatible | incompatible | verdict_agree |
| chillers/refrigerant | propane | carbondioxide | incompatible | incompatible | verdict_agree |
| chillers/refrigerant | propane | isobutane | incompatible | incompatible | verdict_agree |
| chillers/refrigerant | propane | r134a | incompatible | incompatible | verdict_agree |
| chillers/refrigerant | carbondioxide | ammonia | incompatible | incompatible | verdict_agree |
| chillers/refrigerant | carbondioxide | propane | incompatible | incompatible | verdict_agree |
| chillers/refrigerant | carbondioxide | isobutane | incompatible | incompatible | verdict_agree |
| chillers/refrigerant | carbondioxide | r134a | incompatible | incompatible | verdict_agree |
| chillers/refrigerant | isobutane | ammonia | incompatible | incompatible | verdict_agree |
| chillers/refrigerant | isobutane | propane | incompatible | incompatible | verdict_agree |
| chillers/refrigerant | isobutane | carbondioxide | incompatible | incompatible | verdict_agree |
| chillers/refrigerant | isobutane | r134a | incompatible | incompatible | verdict_agree |
| chillers/refrigerant | r134a | ammonia | incompatible | incompatible | verdict_agree |
| chillers/refrigerant | r134a | propane | incompatible | incompatible | verdict_agree |
| chillers/refrigerant | r134a | carbondioxide | incompatible | incompatible | verdict_agree |
| chillers/refrigerant | r134a | isobutane | incompatible | incompatible | verdict_agree |
| chillers/expansion valve | electronic | thermostatic | incompatible | incompatible | verdict_agree |
| chillers/expansion valve | electronic | automatic | compatible | compatible | verdict_agree |
| chillers/expansion valve | electronic | float | incompatible | incompatible | verdict_agree |
| chillers/expansion valve | electronic | fixed | incompatible | incompatible | verdict_agree |
| chillers/expansion valve | thermostatic | electronic | length | incompatible | length_stop_excluded_from_verdicts |
| chillers/expansion valve | thermostatic | automatic | compatible | compatible | verdict_agree |
| chillers/expansion valve | thermostatic | float | incompatible | incompatible | verdict_agree |
| chillers/expansion valve | thermostatic | fixed | incompatible | incompatible | verdict_agree |
| chillers/expansion valve | automatic | electronic | compatible | compatible | verdict_agree |
| chillers/expansion valve | automatic | thermostatic | compatible | compatible | verdict_agree |
| chillers/expansion valve | automatic | float | compatible | compatible | verdict_agree |
| chillers/expansion valve | automatic | fixed | incompatible | incompatible | verdict_agree |
| chillers/expansion valve | float | electronic | incompatible | incompatible | verdict_agree |
| chillers/expansion valve | float | thermostatic | incompatible | incompatible | verdict_agree |
| chillers/expansion valve | float | automatic | length | compatible | length_stop_excluded_from_verdicts |
| chillers/expansion valve | float | fixed | incompatible | incompatible | verdict_agree |
| chillers/expansion valve | fixed | electronic | incompatible | incompatible | verdict_agree |
| chillers/expansion valve | fixed | thermostatic | incompatible | incompatible | verdict_agree |
| chillers/expansion valve | fixed | automatic | incompatible | incompatible | verdict_agree |
| chillers/expansion valve | fixed | float | incompatible | incompatible | verdict_agree |
| chillers/condenser cooling | aircooled | groundcoupled | incompatible | incompatible | verdict_agree |
| chillers/condenser cooling | aircooled | seawatercooled | incompatible | incompatible | verdict_agree |
| chillers/condenser cooling | aircooled | rivercooled | incompatible | incompatible | verdict_agree |
| chillers/condenser cooling | aircooled | towercooled | incompatible | incompatible | verdict_agree |
| chillers/condenser cooling | groundcoupled | aircooled | incompatible | incompatible | verdict_agree |
| chillers/condenser cooling | groundcoupled | seawatercooled | incompatible | incompatible | verdict_agree |
| chillers/condenser cooling | groundcoupled | rivercooled | incompatible | incompatible | verdict_agree |
| chillers/condenser cooling | groundcoupled | towercooled | incompatible | incompatible | verdict_agree |
| chillers/condenser cooling | seawatercooled | aircooled | incompatible | incompatible | verdict_agree |
| chillers/condenser cooling | seawatercooled | groundcoupled | incompatible | incompatible | verdict_agree |
| chillers/condenser cooling | seawatercooled | rivercooled | incompatible | incompatible | verdict_agree |
| chillers/condenser cooling | seawatercooled | towercooled | incompatible | incompatible | verdict_agree |
| chillers/condenser cooling | rivercooled | aircooled | incompatible | incompatible | verdict_agree |
| chillers/condenser cooling | rivercooled | groundcoupled | incompatible | incompatible | verdict_agree |
| chillers/condenser cooling | rivercooled | seawatercooled | incompatible | incompatible | verdict_agree |
| chillers/condenser cooling | rivercooled | towercooled | incompatible | incompatible | verdict_agree |
| chillers/condenser cooling | towercooled | aircooled | incompatible | incompatible | verdict_agree |
| chillers/condenser cooling | towercooled | groundcoupled | incompatible | incompatible | verdict_agree |
| chillers/condenser cooling | towercooled | seawatercooled | incompatible | incompatible | verdict_agree |
| chillers/condenser cooling | towercooled | rivercooled | incompatible | incompatible | verdict_agree |
| pumps/pump state | stopped | priming | incompatible | incompatible | verdict_agree |
| pumps/pump state | stopped | starting | incompatible | incompatible | verdict_agree |
| pumps/pump state | stopped | running | incompatible | incompatible | verdict_agree |
| pumps/pump state | stopped | coasting | incompatible | incompatible | verdict_agree |
| pumps/pump state | priming | stopped | incompatible | incompatible | verdict_agree |
| pumps/pump state | priming | starting | compatible | incompatible | verdict_disagree |
| pumps/pump state | priming | running | incompatible | incompatible | verdict_agree |
| pumps/pump state | priming | coasting | incompatible | incompatible | verdict_agree |
| pumps/pump state | starting | stopped | incompatible | incompatible | verdict_agree |
| pumps/pump state | starting | priming | compatible | compatible | verdict_agree |
| pumps/pump state | starting | running | incompatible | incompatible | verdict_agree |
| pumps/pump state | starting | coasting | incompatible | incompatible | verdict_agree |
| pumps/pump state | running | stopped | incompatible | incompatible | verdict_agree |
| pumps/pump state | running | priming | compatible | incompatible | verdict_disagree |
| pumps/pump state | running | starting | incompatible | incompatible | verdict_agree |
| pumps/pump state | running | coasting | incompatible | incompatible | verdict_agree |
| pumps/pump state | coasting | stopped | incompatible | incompatible | verdict_agree |
| pumps/pump state | coasting | priming | incompatible | incompatible | verdict_agree |
| pumps/pump state | coasting | starting | incompatible | incompatible | verdict_agree |
| pumps/pump state | coasting | running | incompatible | incompatible | verdict_agree |
| pumps/seal material | viton | teflon | incompatible | incompatible | verdict_agree |
| pumps/seal material | viton | nitrile | incompatible | incompatible | verdict_agree |
| pumps/seal material | viton | graphite | incompatible | incompatible | verdict_agree |
| pumps/seal material | viton | kalrez | incompatible | incompatible | verdict_agree |
| pumps/seal material | teflon | viton | incompatible | incompatible | verdict_agree |
| pumps/seal material | teflon | nitrile | incompatible | incompatible | verdict_agree |
| pumps/seal material | teflon | graphite | incompatible | incompatible | verdict_agree |
| pumps/seal material | teflon | kalrez | incompatible | incompatible | verdict_agree |
| pumps/seal material | nitrile | viton | incompatible | incompatible | verdict_agree |
| pumps/seal material | nitrile | teflon | incompatible | incompatible | verdict_agree |
| pumps/seal material | nitrile | graphite | incompatible | incompatible | verdict_agree |
| pumps/seal material | nitrile | kalrez | incompatible | incompatible | verdict_agree |
| pumps/seal material | graphite | viton | incompatible | incompatible | verdict_agree |
| pumps/seal material | graphite | teflon | incompatible | incompatible | verdict_agree |
| pumps/seal material | graphite | nitrile | incompatible | incompatible | verdict_agree |
| pumps/seal material | graphite | kalrez | incompatible | incompatible | verdict_agree |
| pumps/seal material | kalrez | viton | incompatible | incompatible | verdict_agree |
| pumps/seal material | kalrez | teflon | incompatible | incompatible | verdict_agree |
| pumps/seal material | kalrez | nitrile | incompatible | incompatible | verdict_agree |
| pumps/seal material | kalrez | graphite | incompatible | incompatible | verdict_agree |
| pumps/drive coupling | rigid | flexible | incompatible | incompatible | verdict_agree |
| pumps/drive coupling | rigid | fluid | incompatible | incompatible | verdict_agree |
| pumps/drive coupling | rigid | eddycurrent | incompatible | incompatible | verdict_agree |
| pumps/drive coupling | rigid | clutch | incompatible | compatible | verdict_disagree |
| pumps/drive coupling | flexible | rigid | incompatible | incompatible | verdict_agree |
| pumps/drive coupling | flexible | fluid | compatible | compatible | verdict_agree |
| pumps/drive coupling | flexible | eddycurrent | compatible | compatible | verdict_agree |
| pumps/drive coupling | flexible | clutch | compatible | compatible | verdict_agree |
| pumps/drive coupling | fluid | rigid | incompatible | incompatible | verdict_agree |
| pumps/drive coupling | fluid | flexible | compatible | compatible | verdict_agree |
| pumps/drive coupling | fluid | eddycurrent | incompatible | incompatible | verdict_agree |
| pumps/drive coupling | fluid | clutch | length | compatible | length_stop_excluded_from_verdicts |
| pumps/drive coupling | eddycurrent | rigid | incompatible | incompatible | verdict_agree |
| pumps/drive coupling | eddycurrent | flexible | compatible | compatible | verdict_agree |
| pumps/drive coupling | eddycurrent | fluid | incompatible | incompatible | verdict_agree |
| pumps/drive coupling | eddycurrent | clutch | compatible | unsure | verdict_disagree |
| pumps/drive coupling | clutch | rigid | incompatible | incompatible | verdict_agree |
| pumps/drive coupling | clutch | flexible | compatible | length | length_stop_excluded_from_verdicts |
| pumps/drive coupling | clutch | fluid | incompatible | incompatible | verdict_agree |
| pumps/drive coupling | clutch | eddycurrent | incompatible | compatible | verdict_disagree |
| pumps/bearing type | magnetic | roller | incompatible | incompatible | verdict_agree |
| pumps/bearing type | magnetic | sleeve | incompatible | error | length_stop_excluded_from_verdicts |
| pumps/bearing type | magnetic | hydrostatic | incompatible | incompatible | verdict_agree |
| pumps/bearing type | magnetic | ball | incompatible | incompatible | verdict_agree |
| pumps/bearing type | roller | magnetic | incompatible | incompatible | verdict_agree |
| pumps/bearing type | roller | sleeve | incompatible | incompatible | verdict_agree |
| pumps/bearing type | roller | hydrostatic | incompatible | incompatible | verdict_agree |
| pumps/bearing type | roller | ball | incompatible | incompatible | verdict_agree |
| pumps/bearing type | sleeve | magnetic | incompatible | incompatible | verdict_agree |
| pumps/bearing type | sleeve | roller | incompatible | incompatible | verdict_agree |
| pumps/bearing type | sleeve | hydrostatic | compatible | compatible | verdict_agree |
| pumps/bearing type | sleeve | ball | incompatible | incompatible | verdict_agree |
| pumps/bearing type | hydrostatic | magnetic | incompatible | incompatible | verdict_agree |
| pumps/bearing type | hydrostatic | roller | incompatible | incompatible | verdict_agree |
| pumps/bearing type | hydrostatic | sleeve | compatible | compatible | verdict_agree |
| pumps/bearing type | hydrostatic | ball | incompatible | incompatible | verdict_agree |
| pumps/bearing type | ball | magnetic | incompatible | incompatible | verdict_agree |
| pumps/bearing type | ball | roller | incompatible | incompatible | verdict_agree |
| pumps/bearing type | ball | sleeve | incompatible | incompatible | verdict_agree |
| pumps/bearing type | ball | hydrostatic | incompatible | incompatible | verdict_agree |
| pumps/casing mount | foot | centerline | incompatible | incompatible | verdict_agree |
| pumps/casing mount | foot | flange | incompatible | incompatible | verdict_agree |
| pumps/casing mount | foot | pedestal | incompatible | incompatible | verdict_agree |
| pumps/casing mount | foot | inline | incompatible | incompatible | verdict_agree |
| pumps/casing mount | centerline | foot | incompatible | incompatible | verdict_agree |
| pumps/casing mount | centerline | flange | compatible | incompatible | verdict_disagree |
| pumps/casing mount | centerline | pedestal | incompatible | incompatible | verdict_agree |
| pumps/casing mount | centerline | inline | length | compatible | length_stop_excluded_from_verdicts |
| pumps/casing mount | flange | foot | incompatible | incompatible | verdict_agree |
| pumps/casing mount | flange | centerline | incompatible | incompatible | verdict_agree |
| pumps/casing mount | flange | pedestal | incompatible | incompatible | verdict_agree |
| pumps/casing mount | flange | inline | incompatible | incompatible | verdict_agree |
| pumps/casing mount | pedestal | foot | incompatible | incompatible | verdict_agree |
| pumps/casing mount | pedestal | centerline | incompatible | incompatible | verdict_agree |
| pumps/casing mount | pedestal | flange | incompatible | incompatible | verdict_agree |
| pumps/casing mount | pedestal | inline | incompatible | incompatible | verdict_agree |
| pumps/casing mount | inline | foot | incompatible | incompatible | verdict_agree |
| pumps/casing mount | inline | centerline | unsure | length | length_stop_excluded_from_verdicts |
| pumps/casing mount | inline | flange | incompatible | incompatible | verdict_agree |
| pumps/casing mount | inline | pedestal | incompatible | incompatible | verdict_agree |
