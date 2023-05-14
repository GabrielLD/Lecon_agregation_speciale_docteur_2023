tous_les_programmes = [
    'puits_quantique', 
    'oscillateur_amorti',
    'van_der_waals',
    'rlc_serie_force',
    'rlc_serie_declin',
    'loi_de_planck',
    'fentes_young',
    'propagation_son',
    'propagation_onde',
    'interference_elementaire',
    'diffraction_N_fentes',
    'ecoulement_couette',
    'poiseuille',
    'echantillonage_filtrage',
    'portrait_de_phase',
    'propagation_avec_dispersion',
    'effet_tunnel',
    'diffusion_particules',
    'orbites_kepler'
]

tous_les_programmes = list(sorted(tous_les_programmes))

if  __name__=='__main__':
    for name in tous_les_programmes:
        print(name)
