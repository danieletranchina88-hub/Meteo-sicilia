def calculate_front_eta(user_lat: float, user_lon: float, front_lines: list, front_speed_kmh: float, radius_km: float = 20.0):
    """
    Calcola l'orario stimato di arrivo (ETA) di un fronte.
    """
    import math
    
    # Semplice mock di geometria
    # In produzione si userebbe geopandas/shapely per intersezione
    from shapely.geometry import Point, LineString
    
    user_point = Point(user_lon, user_lat)
    
    min_dist_deg = float('inf')
    closest_front = None
    
    for front in front_lines:
        line = LineString(front['coordinates'])
        dist = user_point.distance(line)
        if dist < min_dist_deg:
            min_dist_deg = dist
            closest_front = front
            
    # Approssimazione 1 deg ~ 111 km
    dist_km = min_dist_deg * 111.0
    
    if dist_km <= radius_km:
        return {"status": "imminent", "eta_hours": 0, "message": "Il fronte sta attraversando la tua posizione ora!"}
        
    if closest_front and front_speed_kmh > 0:
        eta_hours = dist_km / front_speed_kmh
        return {"status": "approaching", "eta_hours": round(eta_hours, 1), "distance_km": round(dist_km, 1)}
        
    return {"status": "clear", "message": "Nessun fronte in avvicinamento."}
