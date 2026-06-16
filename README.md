Versión: v13_fix_km_final_utm19s_3d

# Generador de secciones transversales desde KMZ/KML - v12

Aplicación Streamlit para crear secciones transversales de cauce desde un archivo KMZ/KML con eje de cauce y curvas de nivel.

## Cambios v12


- Corrige la selección automática de secciones para modelación: el criterio principal vuelve a ser que la sección tenga ambas riberas definidas.
- Mantiene la vista `Longitudinal seleccionado 3D` de la versión anterior.
- La calidad previa `DEBIL` se informa como advertencia, pero no descarta automáticamente una sección si cumple con ambas riberas.
- Se separa la exigencia de cota en eje para modelación, quedando desactivada por defecto.

## Cambios v11 mantenidos

- Agrega selección explícita de sistema de coordenadas UTM:
  - datum;
  - huso UTM;
  - hemisferio;
  - EPSG manual opcional.
- Valor por defecto para Región de Coquimbo:

```text
Datum: WGS84
Huso UTM: 19
Hemisferio: S
CRS activo: EPSG:32719 — WGS 84 / UTM zone 19S
```

- Las tablas y salidas de cálculo trabajan en coordenadas métricas UTM:
  - Este UTM;
  - Norte UTM;
  - offset transversal;
  - progresiva sobre eje.
- KMZ/KML y GeoJSON se exportan en lon/lat WGS84 para compatibilidad con Google Earth y visores SIG.

## Cambios v10 mantenidos

- Nueva pestaña `Longitudinal seleccionado 3D`.
- Vista 3D con:
  - X = progresiva sobre eje;
  - Y = offset transversal;
  - Z = cota del terreno.
- Azul = secciones seleccionadas para modelación.
- Rojo = km sin sección válida que deben cargarse manualmente.

## Cambios v9 mantenidos

- Vista en planta limpia para evitar enredo de líneas.
- Secciones correctas en azul.
- Secciones descartadas/carga manual en rojo.
- KMZ limpio para Google Earth:
  - `14_kmz_limpio_modelacion_azul_rojo.kmz`.

## Cambios v8 mantenidos

- Filtro de secciones aptas para modelación hidráulica.
- Una sección solo queda seleccionada si tiene ambas riberas definidas respecto del eje del cauce.
- Las secciones insuficientes quedan marcadas como `CARGA_MANUAL`.
- El Excel incluye planillas adicionales con solo las secciones aptas para modelación.
- El perfil longitudinal de modelación destaca en rojo los km que no tienen sección válida y deben cargarse manualmente.

## Ejecución local

```bash
pip install -r requirements.txt
streamlit run app_secciones_kmz.py
```

En Windows también puede usarse:

```text
ejecutar_app_windows.bat
```

## Streamlit Cloud

Main file path:

```text
app_secciones_kmz.py
```

## Nota de coordenadas

El estándar KML/KMZ almacena coordenadas geográficas en longitud/latitud WGS84. La aplicación lee esas coordenadas y las transforma al sistema UTM seleccionado para calcular distancias, anchos, offsets, progresivas, DXF y tablas. Para Google Earth se vuelve a exportar en WGS84 geográfico.

## Advertencia técnica

La geometría construida desde curvas de nivel es preliminar. Para diseño hidráulico, socavación o transporte de sedimentos debe complementarse con topografía de terreno, MDE/LiDAR, rugosidad, caudales de diseño y granulometría.
