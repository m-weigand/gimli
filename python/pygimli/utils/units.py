#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collection of frequently used physical quantities with units and
there abbreviations."""

import pygimli as pg

quants = {
    'rhoa': {
        'name': 'Apparent resistivity',
        'unit': '$\Omega$m',
        'ger': 'scheinbarer spez. elektr. Widerstand',
        'cMap': 'Spectral_r',
    },
    'res': {
        'name': 'Resistivity',
        'unit': '$\Omega$m',
        'ger': 'spez. elektr. Widerstand',
        'cMap': 'Spectral_r',
    },
    'ip': {
        'name': 'Phase',
        'unit': 'mrad',
        'ger': 'Phasenwinkel',
        'cMap': 'viridis',
    },
   'ip_n': {
        'name': 'neg. Phase',
        'unit': 'mrad',
        'ger': 'neg. Phasenwinkel',
        'cMap': 'viridis',
    }, 
    'ipa': {
        'name': 'Apparent phase',
        'unit': 'mrad',
        'ger': 'scheinbarer Phasenwinkel',
        'cMap': 'viridis',
    },
    'ipa_n': {
        'name': 'neg. Apparent phase',
        'unit': 'mrad',
        'ger': 'neg. scheinbarer Phasenwinkel',
        'cMap': 'viridis',
    },
    'va': {
        'name': 'Apparent velocity',
        'unit': 'm/s',
        'ger': 'Scheingeschwindigkeit',
    },
    'vel': {
        'name': 'Velocity',
        'unit': 'm/s',
        'ger': 'Geschwindigkeit',
    },
    'as': {
        'name': 'Apparent slowness',
        'unit': 's/m',
        'ger': 'Scheinlangsamkeit',
    },
    'slo': {
        'name': 'Slowness',
        'unit': 's/m',
        'ger': 'Slowness',
    },
}

rc = {
    'lang': 'english',
    'unitStyle': 2,  # quantity (unit)
}

def quantity(name):
    """ """
    quantity = None
    
    if name.lower() not in quants:
        for k, v in quants.items():
            if v['name'].lower() == name.lower():
                quantity = v
                break
    else:
        quantity = quants[name]
    return quantity

def cMap(name):
    """Return default colormap for physical quantity name."""
    q = quantity(name)
    if q is None:
        pg.warning('No information about quantity name', name)
        return 'vididis'
    return q.get('cMap', 'vididis')


def unit(name, unit='auto'):
    """ Return the name of a physical quantity with its unit.

    TODO
    ----
        * example
        * localization

    Parameters
    ----------
    """
    q = quantity(name)

    if unit == 'auto' and q is None:
        ## fall back if the name is given instead of the abbreviation
        print(quants)
        pg.error('Please give abbreviation or full name '
                 'for the quantity name: {0}'.format(name))
    else:
        if rc['lang'] == 'german':
            name = q['ger']
        else:
            name = q['name']

        if unit == 'auto':
            unit = q['unit']

    if rc['unitStyle'] == 1 or rc['lang'] == 'german':
        return '{0} in {1}'.format(name, unit)
    else:
        return '{0} ({1})'.format(name, unit)
