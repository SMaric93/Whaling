# Destination Ontology

## Overview
- **662** unique ground/route labels
- **6** basins
- **72** theaters
- **491** singletons (≤2 voyages) collapsed to theater level

## Basin Summary

| basin       |   n_labels |   n_voyages |   n_theaters |
|:------------|-----------:|------------:|-------------:|
| Pacific     |         88 |        4314 |            8 |
| Atlantic    |        209 |        2940 |            8 |
| Unknown     |        159 |         776 |            1 |
| Multi-Ocean |        170 |         415 |           50 |
| Indian      |         27 |         334 |            3 |
| Arctic      |          9 |          95 |            2 |

## Top 20 Theaters

|                                                           |   n_labels |   n_voyages |
|:----------------------------------------------------------|-----------:|------------:|
| ('Pacific', 'Pacific')                                    |         62 |        4140 |
| ('Atlantic', 'Atlantic')                                  |        137 |        2155 |
| ('Unknown', 'Unknown')                                    |        159 |         776 |
| ('Atlantic', 'Brazil')                                    |         35 |         451 |
| ('Indian', 'Indian Ocean')                                |         13 |         266 |
| ('Atlantic', 'Gulf of Mexico')                            |         12 |         117 |
| ('Multi-Ocean', 'Atlantic + Indian Ocean')                |         18 |         103 |
| ('Atlantic', 'Patagonia/Falklands')                       |          9 |         100 |
| ('Pacific', 'Japan Grounds')                              |          4 |          85 |
| ('Arctic', 'Hudson Bay')                                  |          4 |          71 |
| ('Multi-Ocean', 'Indian Ocean + Pacific')                 |         18 |          69 |
| ('Atlantic', 'Western Islands/Azores')                    |          3 |          54 |
| ('Atlantic', 'Cape Verde')                                |          4 |          43 |
| ('Multi-Ocean', 'Atlantic + Pacific')                     |         10 |          40 |
| ('Indian', 'Africa/Madagascar')                           |         10 |          36 |
| ('Multi-Ocean', 'Arctic + North Pacific + Pacific')       |          9 |          33 |
| ('Pacific', 'NW Coast')                                   |          6 |          32 |
| ('Indian', 'Desolation/Kerguelen')                        |          4 |          32 |
| ('Arctic', 'Arctic')                                      |          5 |          24 |
| ('Multi-Ocean', 'Indian Ocean + North Pacific + Pacific') |          6 |          23 |

## Hierarchy
```
basin → theater → major_ground → local_ground
```

Singletons are collapsed: `ground_for_model` = theater for rare labels.