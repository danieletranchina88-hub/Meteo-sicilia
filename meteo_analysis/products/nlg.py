def generate_bulletin(front_type: str, prob_thunderstorm: str, hail_threat: str, t_trend: str) -> str:
    """
    Genera un bollettino testuale basato sulle condizioni meteorologiche.
    """
    testo = "Previsione automatica per la zona: \n"
    
    if front_type == "freddo":
        testo += "E' in arrivo un fronte freddo che determinerà un " + t_trend + " delle temperature. "
    elif front_type == "caldo":
        testo += "Un fronte caldo sta portando copertura nuvolosa stratificata e un " + t_trend + " termico. "
    elif front_type == "stazionario":
        testo += "Un fronte stazionario insiste sull'area, portando precipitazioni persistenti. "
        
    if prob_thunderstorm == "alta":
        testo += "Attenzione: c'è un'elevata probabilità di sviluppo di temporali intensi. "
        if hail_threat == "alto":
            testo += "Possibilità di grandinate significative e potenziali supercelle. "
            
    return testo
