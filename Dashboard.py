import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import glob
import os
import numpy as np
import math
from scipy.stats import linregress
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster
import geopandas as gpd
import string
import re

# Set page config
st.set_page_config(
    page_title="Nepal Lakes WSE Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
    <style>
    .stApp {
        background-color: white;
    }
    
    /* Remove default streamlit padding */
    .main .block-container {
        padding-top: 0rem;
        padding-left: 0rem;
        padding-right: 0rem;
        max-width: 100%;
    }

    /* --- Clean Top Title & Nav --- */
    .top-title {
        color: #2d3748;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0;
        padding: 16px 0 8px 0;
        display: inline-block;
        vertical-align: middle;
    }

    .nav-buttons-container {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        align-items: center;
        margin: 8px 0 20px 0;
        padding: 0;
    }

    .nav-btn {
        background-color: #f7fafc;
        border: 1px solid #e2e8f0;
        padding: 8px 16px;
        border-radius: 6px;
        cursor: pointer;
        font-size: 0.875rem;
        font-weight: 500;
        transition: all 0.2s ease;
        color: #2d3748;
        text-align: center;
        min-width: 100px;
        white-space: nowrap;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }

    .nav-btn:hover {
        background-color: #edf2f7;
        border-color: #a0aec0;
        transform: translateY(-1px);
        box-shadow: 0 2px 6px rgba(0,0,0,0.12);
    }

    .nav-btn.active {
        background-color: #3182ce !important;
        color: white !important;
        font-weight: 600;
        border-color: #2c5aa0 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.15);
    }

    .contact-btn {
        background-color: #e53e3e !important;
        color: white !important;
        font-weight: 600;
        border-color: #c53030 !important;
    }

    .contact-btn:hover {
        background-color: #c53030 !important;
        color: white !important;
        border-color: #a02727 !important;
    }

    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'Dashboard'

# --- Clean Top Title ---
st.markdown('<h1 class="top-title">🌊 Nepal SWOT-based Water Monitoring Framework</h1>', unsafe_allow_html=True)

# --- Navigation Buttons ---
st.markdown('<div class="nav-buttons-container">', unsafe_allow_html=True)

cols = st.columns(4)
button_labels = ["📊 Dashboard", "❓ How to Use", "🛰️ About SWOT", "📧 Contact Us"]
button_keys = ["nav_dashboard", "nav_howto", "nav_about", "nav_contact"]

for i, col in enumerate(cols):
    with col:
        current_label = button_labels[i].split(" ", 1)[1] if " " in button_labels[i] else button_labels[i]
        is_active = st.session_state.page == current_label
        btn_class = "nav-btn active" if is_active else "nav-btn"
        if i == 3:  # Contact button
            btn_class += " contact-btn"
        
        if st.button(button_labels[i], key=button_keys[i], use_container_width=True):
            if current_label == "Dashboard":
                st.session_state.page = 'Dashboard'
            elif current_label == "How to Use":
                st.session_state.page = 'How to Use'
            elif current_label == "About SWOT":
                st.session_state.page = 'About SWOT'
            elif current_label == "Contact Us":
                st.session_state.page = 'Contact Us'
            st.rerun()

st.markdown('</div>', unsafe_allow_html=True)  # Close buttons container

st.markdown("---")  # Separator

# --- Sidebar - Lake selection ---
with st.sidebar:
    st.title("🌊 Data Controls")
    st.markdown("---")
    if st.session_state.page == 'Dashboard':
        if 'df' in st.session_state and st.session_state.df is not None and 'Lake' in st.session_state.df.columns:
            selected_lake = st.selectbox(
                "**Select station from drop-down:**",
                st.session_state.df["Lake"].unique(),
                index=st.session_state.df["Lake"].unique().tolist().index(st.session_state.selected_lake) if 'selected_lake' in st.session_state and st.session_state.selected_lake in st.session_state.df["Lake"].unique() else 0
            )
            st.session_state.selected_lake = selected_lake
            st.markdown(f"### Selected Station: {selected_lake}")
        else:
            st.info("Load data to select a lake station")

# Load data — preserve raw names
@st.cache_data
def load_data():
    folder = "WSE_Data"
    if not os.path.exists(folder):
        st.error("📁 Data folder not found: 'WSE_Data'. Please create it.")
        return None
    all_files = glob.glob(os.path.join(folder, "*.csv"))
    if not all_files:
        st.error("❌ No CSV files found in 'WSE_Data'. Add your CSV files.")
        return None

    df_list = []
    lake_coordinates = {}
    successful_files = 0

    for file in all_files:
        try:
            temp = pd.read_csv(file)
            filename = os.path.basename(file)
            date_col = next((c for c in ['datetime', 'Date'] if c in temp.columns), None)
            wse_col = next((c for c in ['wse', 'WSE'] if c in temp.columns), None)
            if not date_col or not wse_col:
                st.warning(f"⚠️ Skipping {filename}: Missing 'datetime'/Date or wse/WSE")
                continue
            temp["Date"] = pd.to_datetime(temp[date_col])
            temp["WSE"] = temp[wse_col]

            # Preserve raw name — no cleaning
            lake_name = filename.replace("_WSE_Temporal_Analysis_data.csv", "").replace(".csv", "").replace("_", " ")
            temp["Lake"] = lake_name

            # Generate random coords (for now)
            np.random.seed(hash(lake_name) % 1000)
            lat = 27.5 + np.random.uniform(-2, 2)  # Nepal coordinates
            lon = 84.5 + np.random.uniform(-2, 2)
            lake_coordinates[lake_name] = (lat, lon)

            df_list.append(temp)
            successful_files += 1
        except Exception as e:
            st.warning(f"⚠️ Error reading {filename}: {str(e)}")

    if not df_list:
        st.error("❌ No valid data loaded.")
        return None

    try:
        df = pd.concat(df_list, ignore_index=True)
        st.session_state.lake_coordinates = lake_coordinates
        return df
    except Exception as e:
        st.error(f"❌ Error combining: {str(e)}")
        return None

# Functions
def flag_outliers_iqr(data, colname):
    Q1 = np.percentile(data[colname], 25)
    Q3 = np.percentile(data[colname], 75)
    IQR = Q3 - Q1
    return (data[colname] < Q1 - 1.5 * IQR) | (data[colname] > Q3 + 1.5 * IQR)

def compute_trend(x, y):
    if len(x) < 2:
        return None
    try:
        x_num = np.array([d.toordinal() for d in x])
        return linregress(x_num, y)
    except:
        return None

# Enhanced name cleaning function
def clean_name(name):
    # Convert to string and lowercase
    name = str(name).lower()
    
    # Remove all punctuation and special characters
    name = re.sub(r'[^\w\s]', '', name)
    
    # Replace multiple spaces with single space
    name = re.sub(r'\s+', ' ', name)
    
    # Remove common prefixes
    prefixes = ['lake', 'pond', 'reservoir', 'waterbody', 'tal', 'pokhari']
    for prefix in prefixes:
        if name.startswith(prefix):
            name = name.replace(prefix, '', 1).strip()
    
    # Remove any remaining whitespace
    name = name.strip()
    
    return name

# Enhanced Map — Name-based matching and click events
def create_enhanced_map(selected_lake=None):
    try:
        map_center = [28.0, 84.0]  # Default: Nepal center
        zoom_start = 7
        target_lat, target_lon = None, None

        # Load lakes shapefile
        try:
            lakes_gdf = gpd.read_file("Shapefiles/SWOT_Lakes.shp")
            if isinstance(lakes_gdf, gpd.GeoSeries):
                lakes_gdf = gpd.GeoDataFrame({"geometry": lakes_gdf})

            # Standardize column name for lake names
            possible_names = ['lake_name', 'Lake_Name', 'LAKE_NAME', 'name', 'Name', 'NAME', 'GNIS_NAME', 'gnis_name']
            lake_name_col = next((col for col in possible_names if col in lakes_gdf.columns), None)
            if lake_name_col:
                lakes_gdf = lakes_gdf.rename(columns={lake_name_col: 'lake_name'})
            else:
                for col in lakes_gdf.columns:
                    if lakes_gdf[col].dtype == 'object' and col != 'geometry':
                        lakes_gdf = lakes_gdf.rename(columns={col: 'lake_name'})
                        break
                else:
                    lakes_gdf['lake_name'] = "Lake " + lakes_gdf.index.astype(str)

        except Exception as e:
            st.warning(f"Could not load lakes shapefile: {str(e)}")
            lakes_gdf = None

        # Try to find and center on selected lake
        if selected_lake and lakes_gdf is not None:
            selected_clean = clean_name(selected_lake)
            best_match = None
            best_score = 0
            
            for idx, row in lakes_gdf.iterrows():
                lake_name = row['lake_name']
                lake_clean = clean_name(lake_name)
                
                # Strategy 1: Exact match after cleaning
                if selected_clean == lake_clean:
                    best_match = row
                    best_score = 100
                    break
                
                # Strategy 2: Partial match (selected is contained in lake name)
                if selected_clean in lake_clean:
                    score = len(selected_clean) / len(lake_clean) * 90
                    if score > best_score:
                        best_match = row
                        best_score = score
                
                # Strategy 3: Lake name is contained in selected
                elif lake_clean in selected_clean:
                    score = len(lake_clean) / len(selected_clean) * 80
                    if score > best_score:
                        best_match = row
                        best_score = score
                
                # Strategy 4: Fuzzy matching for similar names
                elif (len(selected_clean) > 3 and len(lake_clean) > 3 and 
                      selected_clean[:4] == lake_clean[:4]):
                    score = 70
                    if score > best_score:
                        best_match = row
                        best_score = score

            if best_match is not None:
                geom = best_match.geometry.centroid
                target_lat, target_lon = geom.y, geom.x
                map_center = [target_lat, target_lon]
                zoom_start = 10
            else:
                st.sidebar.info(f"🔍 No matching lake found for '{selected_lake}'. Showing all lakes.")

        m = folium.Map(location=map_center, zoom_start=zoom_start, tiles=None, control_scale=True)

        # Add tile layers
        folium.TileLayer(tiles='OpenStreetMap', name='OpenStreetMap', attr='OSM').add_to(m)
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Esri World Imagery'
        ).add_to(m)
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Esri World Topo'
        ).add_to(m)

        # Nepal Boundary Layer
        nepal_layer = folium.FeatureGroup(name='Nepal Boundary', show=True)
        try:
            nepal_gdf = gpd.read_file("Shapefiles/Nepal Outer Boundary.shp")
            if isinstance(nepal_gdf, gpd.GeoSeries):
                nepal_gdf = gpd.GeoDataFrame({"geometry": nepal_gdf})
            nepal_gdf['geometry'] = nepal_gdf['geometry'].simplify(0.01)
            def style_nepal(feature):
                return {'fillColor': '#8B0000', 'color': '#8B0000', 'weight': 3, 'fillOpacity': 0.2}
            folium.GeoJson(
                nepal_gdf,
                style_function=style_nepal,
                name='Nepal Boundary'
            ).add_to(nepal_layer)
            nepal_layer.add_to(m)
        except Exception as e:
            st.warning(f"Could not load Nepal boundary: {str(e)}")

        # Lakes Layer
        lakes_layer = folium.FeatureGroup(name='Lakes', show=True)

        if lakes_gdf is not None:
            for idx, row in lakes_gdf.iterrows():
                lake_name = row['lake_name']
                centroid = row.geometry.centroid
                lat, lon = centroid.y, centroid.x

                # Check if this lake matches the selected one
                is_selected = False
                if selected_lake:
                    lake_clean = clean_name(lake_name)
                    selected_clean = clean_name(selected_lake)
                    if (selected_clean == lake_clean or 
                        selected_clean in lake_clean or 
                        lake_clean in selected_clean or
                        (len(selected_clean) > 3 and len(lake_clean) > 3 and 
                         selected_clean[:4] == lake_clean[:4])):
                        is_selected = True

                # Define marker appearance
                if is_selected:
                    icon_html = '''
                    <div style="display: inline-block;
                                background-color: rgba(30, 144, 255, 0.8);
                                border: 2px solid red;
                                border-radius: 50%;
                                width: 18px;
                                height: 18px;
                                text-align: center;">
                    </div>
                    '''
                    icon_size = (18, 18)
                    icon_anchor = (9, 9)
                else:
                    icon_html = '''
                    <div style="display: inline-block;
                                background-color: rgba(30, 144, 255, 0.6);
                                border: 1px solid white;
                                border-radius: 50%;
                                width: 12px;
                                height: 12px;
                                text-align: center;">
                    </div>
                    '''
                    icon_size = (12, 12)
                    icon_anchor = (6, 6)

                icon = folium.DivIcon(
                    html=icon_html,
                    icon_size=icon_size,
                    icon_anchor=icon_anchor
                )

                popup_content = f"<b>{lake_name}</b><br><i>{'🔴 SELECTED' if is_selected else '🔵 Normal'}</i>"

                # Create marker with click event
                marker = folium.Marker(
                    location=[lat, lon],
                    popup=popup_content,
                    tooltip=f"{'[SELECTED] ' if is_selected else ''}{lake_name}",
                    icon=icon
                )
                marker.add_to(lakes_layer)

                # Add click event
                marker_id = f"marker_{idx}"
                marker.options['marker_id'] = marker_id
                click_js = f"""
                <script>
                    document.getElementById('{marker_id}').addEventListener('click', function() {{
                        window.parent.postMessage({{
                            type: 'STREAMLIT_FOLIUM_EVENT',
                            data: {{ lake_name: '{lake_name}' }}
                        }}, '*');
                    }});
                </script>
                """
                marker._id = marker_id  # Set marker ID for JavaScript
                m.get_root().html.add_child(folium.Element(click_js))

                # Add label for selected lake
                if is_selected:
                    label_color = "red"
                    label_font_size = "12px"
                    label = folium.Marker(
                        location=[lat + 0.01, lon],
                        icon=folium.DivIcon(
                            html=f'<div style="font-weight: bold; color: {label_color}; font-size: {label_font_size}; background: rgba(255,255,255,0.7); padding: 2px; border-radius: 3px;">{lake_name}</div>',
                            icon_size=(150, 30),
                            icon_anchor=(0, 0)
                        )
                    )
                    label.add_to(lakes_layer)

            lakes_layer.add_to(m)

        # Fly to selected lake
        if target_lat is not None and target_lon is not None:
            fly_to_js = f"""
            <script>
                function flyToSelected() {{
                    var map = document.querySelector('.folium-map')._leaflet_map;
                    if (map) {{
                        map.flyTo([{target_lat}, {target_lon}], 10, {{
                            duration: 1.5,
                            easeLinearity: 0.25
                        }});
                    }}
                }}
                setTimeout(flyToSelected, 800);
            </script>
            """
            m.get_root().html.add_child(folium.Element(fly_to_js))

        # Layer Control
        folium.LayerControl(collapsed=False).add_to(m)
        return m
    except Exception as e:
        st.error(f"Error creating map: {str(e)}")
        return None

# --- Page Rendering ---
if st.session_state.page == 'Dashboard':
    if 'df' not in st.session_state or st.session_state.df is None:
        st.session_state.df = load_data()

    if st.session_state.df is not None:
        st.subheader("Gauging locations over Nepal")
        
        # Render the map and capture click events
        map_output = st_folium(
            create_enhanced_map(st.session_state.get('selected_lake')),
            width=1200,
            height=500,
            returned_objects=['last_clicked', 'data']
        )

        # Handle map click events
        if map_output and 'data' in map_output and map_output['data'] and 'lake_name' in map_output['data']:
            selected_lake_from_map = map_output['data']['lake_name']
            available_lakes = st.session_state.df["Lake"].unique()
            # Try exact match first
            if selected_lake_from_map in available_lakes:
                st.session_state.selected_lake = selected_lake_from_map
                st.rerun()
            else:
                # Try fuzzy matching
                selected_clean = clean_name(selected_lake_from_map)
                best_match = None
                best_score = 0
                for lake in available_lakes:
                    lake_clean = clean_name(lake)
                    if selected_clean == lake_clean:
                        best_match = lake
                        best_score = 100
                        break
                    elif selected_clean in lake_clean:
                        score = len(selected_clean) / len(lake_clean) * 90
                        if score > best_score:
                            best_match = lake
                            best_score = score
                    elif lake_clean in selected_clean:
                        score = len(lake_clean) / len(selected_clean) * 80
                        if score > best_score:
                            best_match = lake
                            best_score = score
                    elif (len(selected_clean) > 3 and len(lake_clean) > 3 and 
                          selected_clean[:4] == lake_clean[:4]):
                        score = 70
                        if score > best_score:
                            best_match = lake
                            best_score = score
                if best_match:
                    st.session_state.selected_lake = best_match
                    st.rerun()
                else:
                    st.warning(f"Selected lake '{selected_lake_from_map}' not found in data. Please select a lake from the dropdown or ensure lake names match.")

        if 'selected_lake' in st.session_state and st.session_state.selected_lake:
            st.subheader(f"WSE Time Series: {st.session_state.selected_lake}")
            lake_data = st.session_state.df[st.session_state.df["Lake"] == st.session_state.selected_lake].copy()
            if not lake_data.empty:
                lake_data = lake_data.sort_values('Date')
                lake_data["is_outlier_wse"] = flag_outliers_iqr(lake_data, "WSE")
                wse_non = lake_data[lake_data["is_outlier_wse"] == False]
                wse_out = lake_data[lake_data["is_outlier_wse"] == True]
                trend_wse = compute_trend(wse_non["Date"], wse_non["WSE"])

                # Create interactive plot with Plotly
                fig = go.Figure()
                
                # Add non-outlier points with connecting line
                fig.add_trace(go.Scatter(
                    x=wse_non["Date"], 
                    y=wse_non["WSE"],
                    mode='lines+markers',
                    name='WSE (m)',
                    marker=dict(
                        color='blue',
                        size=6,
                        line=dict(width=1, color='white')
                    ),
                    line=dict(
                        color='blue',
                        width=2
                    ),
                    hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>WSE</b>: %{y:.2f} m<extra></extra>'
                ))
                
                # Add outlier points
                if not wse_out.empty:
                    fig.add_trace(go.Scatter(
                        x=wse_out["Date"], 
                        y=wse_out["WSE"],
                        mode='markers',
                        name='Outliers',
                        marker=dict(
                            color='red',
                            size=8,
                            symbol='x',
                            line=dict(width=1, color='white')
                        ),
                        hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>WSE</b>: %{y:.2f} m<extra></extra>'
                    ))
                
                # Connect outliers with dashed lines
                if not wse_out.empty:
                    df_sorted = lake_data.sort_values("Date")
                    is_non_outlier = ~df_sorted["is_outlier_wse"]
                    non_outlier_indices = np.where(is_non_outlier)[0]
                    
                    if len(non_outlier_indices) >= 2:
                        for i in range(len(non_outlier_indices) - 1):
                            start_idx = non_outlier_indices[i]
                            end_idx = non_outlier_indices[i + 1]
                            segment = df_sorted.iloc[start_idx:end_idx + 1]
                            outlier_segment = segment[segment["is_outlier_wse"]]
                            
                            if not outlier_segment.empty:
                                # Connect first non-outlier to first outlier
                                first_non_outlier = segment.iloc[0]
                                first_outlier = outlier_segment.iloc[0]
                                fig.add_trace(go.Scatter(
                                    x=[first_non_outlier["Date"], first_outlier["Date"]],
                                    y=[first_non_outlier["WSE"], first_outlier["WSE"]],
                                    mode='lines',
                                    line=dict(color='red', width=1, dash='dash'),
                                    showlegend=False
                                ))
                                
                                # Connect consecutive outliers
                                for j in range(len(outlier_segment) - 1):
                                    curr = outlier_segment.iloc[j]
                                    next_ = outlier_segment.iloc[j + 1]
                                    fig.add_trace(go.Scatter(
                                        x=[curr["Date"], next_["Date"]],
                                        y=[curr["WSE"], next_["WSE"]],
                                        mode='lines',
                                        line=dict(color='red', width=1, dash='dash'),
                                        showlegend=False
                                    ))
                                
                                # Connect last outlier to next non-outlier
                                last_outlier = outlier_segment.iloc[-1]
                                second_non_outlier = segment.iloc[-1]
                                fig.add_trace(go.Scatter(
                                    x=[last_outlier["Date"], second_non_outlier["Date"]],
                                    y=[last_outlier["WSE"], second_non_outlier["WSE"]],
                                    mode='lines',
                                    line=dict(color='red', width=1, dash='dash'),
                                    showlegend=False
                                ))
                
                # Add trend line if available
                if trend_wse:
                    x_vals = wse_non["Date"]
                    x_num = np.array([d.toordinal() for d in x_vals])
                    y_fit = trend_wse.intercept + trend_wse.slope * x_num
                    
                    fig.add_trace(go.Scatter(
                        x=x_vals,
                        y=y_fit,
                        mode='lines',
                        name='Trend',
                        line=dict(
                            color='navy',
                            width=2,
                            dash='dash'
                        ),
                        hovertemplate='<b>Trend</b>: %{y:.2f} m<extra></extra>'
                    ))
                
                # Calculate mean WSE for horizontal line
                mean_wse = wse_non["WSE"].mean()
                
                # Add mean horizontal line for major y-axis (light gray)
                fig.add_hline(y=mean_wse, line_dash="dash", line_color="lightgray", line_width=1, annotation_text="Mean", annotation_position="top right")
                
                # Calculate y-axis range with buffer
                y_min = lake_data["WSE"].min()
                y_max = lake_data["WSE"].max()
                bottom_buffer = 0.10  # 10 cm
                top_buffer = 0.03     # 3 cm
                y_min_pad = y_min - bottom_buffer
                y_max_pad = y_max + top_buffer
                
                # Dynamic y-ticks (major only)
                y_range = y_max - y_min
                base_interval = y_range / 4
                adjusted_interval = math.floor(base_interval * 2) / 2
                major_interval = max(math.floor(adjusted_interval) + 0.5 if adjusted_interval % 1 != 0 else adjusted_interval, 0.5)
                y_min_round = math.floor(y_min_pad / major_interval) * major_interval
                y_max_round = math.ceil(y_max_pad / major_interval) * major_interval
                major_ticks = np.arange(y_min_round, y_max_round + major_interval, major_interval)
                if len(major_ticks) > 6:
                    major_ticks = np.linspace(y_min_round, y_max_round, 6, endpoint=True)
                
                # Generate monthly x-axis ticks
                date_min = lake_data["Date"].min()
                date_max = lake_data["Date"].max()
                # Start from the first day of the month of the minimum date
                start_date = date_min.replace(day=1)
                # End at the first day of the month after the maximum date
                end_date = (date_max + pd.offsets.MonthEnd(0)).replace(day=1) + pd.offsets.MonthBegin(1)
                # Generate monthly ticks
                monthly_ticks = pd.date_range(start=start_date, end=end_date, freq='MS')
                
                # Update layout
                fig.update_layout(
                    title=dict(
                        text=f"Temporal Analysis: {st.session_state.selected_lake}",
                        font=dict(size=16, color='#2d3748', family='Arial, sans-serif', weight='bold'),
                        x=0.5,
                        xanchor='center',
                        pad=dict(t=20)
                    ),
                    xaxis_title=dict(
                        text="Date",
                        font=dict(size=14, color='#2d3748', family='Arial, sans-serif', weight='bold')
                    ),
                    yaxis_title=dict(
                        text="Water Surface Elevation (m)",
                        font=dict(size=14, color='blue', family='Arial, sans-serif', weight='bold')
                    ),
                    xaxis=dict(
                        tickmode='array',
                        tickvals=monthly_ticks,
                        ticktext=[tick.strftime('%Y-%m') for tick in monthly_ticks],
                        tickfont=dict(size=10, color='#2d3748', family='Arial, sans-serif', weight='bold'),
                        tickangle=45,
                        gridcolor='rgba(0,0,0,0.7)',
                        gridwidth=1.2,
                        zeroline=True,
                        zerolinecolor='black',
                        zerolinewidth=1
                    ),
                    yaxis=dict(
                        tickfont=dict(size=10, color='blue', family='Arial, sans-serif', weight='bold'),
                        range=[y_min_round, y_max_round],
                        tickvals=[y_min_round] + major_ticks.tolist() + [y_max_round],
                        ticktext=[f"{y_min_round:.2f}"] + [f"{tick:.2f}" for tick in major_ticks] + [f"{y_max_round:.2f}"],
                        showgrid=True,
                        gridcolor='rgba(200,200,200,0.5)',
                        gridwidth=1.0,
                        zeroline=True,
                        zerolinecolor='black',
                        zerolinewidth=1.2
                    ),
                    hovermode="x unified",
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=-0.35,
                        xanchor="center",
                        x=0.5,
                        font=dict(size=11, family='Arial, sans-serif')
                    ),
                    height=500,
                    template="plotly_white",
                    margin=dict(l=50, r=50, t=80, b=80)
                )
                
                # Add dark horizontal line at minimum y value
                fig.add_hline(
                    y=y_min_round, 
                    line_dash="solid", 
                    line_color="black", 
                    line_width=2,
                    layer="below"
                )
                
                # Add light horizontal line at maximum y value
                fig.add_hline(
                    y=y_max_round, 
                    line_dash="solid", 
                    line_color="lightgray", 
                    line_width=1,
                    layer="below"
                )
                
                # Add small vertical tick marks along the minimum y-axis line for each data point
                unique_dates = sorted(lake_data["Date"].unique())
                tick_height = (y_max_round - y_min_round) * 0.01  # 1% of y-range for tick height
                
                for date in unique_dates:
                    fig.add_shape(
                        type="line",
                        x0=date, y0=y_min_round,
                        x1=date, y1=y_min_round + tick_height,
                        line=dict(color="black", width=1),
                        layer="below"
                    )
                
                # Display the plot
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Summary Statistics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Points", len(lake_data))
                    st.metric("Non-Outlier", len(wse_non))
                with col2:
                    st.metric("Min WSE", f"{lake_data['WSE'].min():.2f} m")
                    st.metric("Max WSE", f"{lake_data['WSE'].max():.2f} m")
                with col3:
                    st.metric("Mean WSE", f"{lake_data['WSE'].mean():.2f} m")
                    if trend_wse:
                        st.metric("Annual Trend", f"{trend_wse.slope*365:.3f} m/yr")

                st.subheader("Download Data")
                csv = lake_data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Data as CSV",
                    data=csv,
                    file_name=f"{st.session_state.selected_lake}_wse.csv",
                    mime="text/csv",
                    key="download-csv"
                )
                
                # Download plot as HTML (Plotly interactive)
                html_buffer = fig.to_html().encode('utf-8')
                st.download_button(
                    label="📊 Download Interactive Plot as HTML",
                    data=html_buffer,
                    file_name=f"{st.session_state.selected_lake}_wse_plot.html",
                    mime="text/html",
                    key="download-plot-html"
                )
        else:
            st.info("Please select a lake from the dropdown of Dashboard to view the temporal analysis.")

elif st.session_state.page == 'How to Use':
    st.title("📖 How to Use This Dashboard")
    st.markdown("""
    ## Getting Started
    
    ### 1. Navigation
    - Use the navigation buttons at the top to switch between different sections
    - **Dashboard**: Main data visualization and analysis
    - **How to Use**: This help guide
    - **About SWOT**: Information about the SWOT mission
    - **Contact Us**: Get in touch with the development team
    
    ### 2. Using the Dashboard
    - **Select a Lake**: Click a lake marker on the map or use the dropdown in the sidebar to choose a specific lake station
    - **View Map**: The interactive map shows all gauging locations across Nepal
    - **Analyze Data**: View interactive time series plots, statistics, and trends for the selected lake
    - **Download**: Export the data as CSV or interactive plots for further analysis
    
    ### 3. Map Features
    - **Multiple Basemaps**: Switch between OpenStreetMap, satellite imagery, and topographic maps
    - **Lake Markers**: Blue markers indicate lake locations, red markers show the currently selected lake
    - **Interactive**: Hover over markers to see lake names; click to select a lake for analysis
    - **Layer Control**: Toggle different map layers on/off (including "Lakes" and "Nepal Boundary")
    
    ### 4. Interactive Data Analysis
    - **Time Series Plot**: Interactive Plotly chart showing water surface elevation over time
    - **Zoom & Pan**: Use mouse to zoom and pan the plot
    - **Range Selector**: Quick buttons to view 1 month, 6 months, 1 year, or all data
    - **Range Slider**: Bottom slider to adjust time window
    - **Hover Information**: Detailed data on hover
    - **Outlier Detection**: Red X marks indicate statistical outliers
    - **Trend Analysis**: Dashed line shows long-term trends
    - **Statistics**: Key metrics including min, max, mean WSE and annual trends
    
    ### 5. Data Requirements
    Your CSV files should be placed in a folder called "WSE_Data" and contain:
    - **Date column**: Named 'datetime' or 'Date'
    - **WSE column**: Named 'wse' or 'WSE' (Water Surface Elevation)
    - **File naming**: Use format "LakeName_WSE_Temporal_Analysis_data.csv"
    
    ## Tips for Best Results
    - Ensure your data files are properly formatted
    - Use consistent date formats (YYYY-MM-DD recommended)
    - Check that WSE values are in meters
    - Ensure lake names in the shapefile match those in the CSV files
    - Use the interactive plot features to explore your data in detail
    """)

elif st.session_state.page == 'About SWOT':
    st.title("🛰️ About SWOT Mission")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ## 🌍 What is SWOT?
        The Surface Water and Ocean Topography (SWOT) mission is a satellite-based project jointly developed by NASA and CNES (the French space agency), with contributions from the Canadian Space Agency (CSA) and the UK Space Agency.
        
        ## 🎯 Mission Objectives
        SWOT is designed to:
        - Provide the first global survey of Earth's surface water
        - Observe the fine details of ocean surface topography
        - Measure how water bodies change over time
        - Improve understanding of water resources worldwide

        ## 🔗 Additional Resources
        For more information about the SWOT mission:
        - [NASA SWOT Mission Website](https://swot.jpl.nasa.gov/)
        - [NOAA SWOT Information](https://www.nesdis.noaa.gov/swot)
        - [CNES SWOT Page](https://www.aviso.altimetry.fr/en/missions/current-missions/swot.html)
        """)
    
    with col2:
        st.info("""
        **Key Facts:**
        - Launched: December 2022
        - Mission Life: 3+ years
        - Accuracy: Centimeter-level
        - Coverage: Global
        """)

    st.markdown("""
    ## 📊 Scientific Applications
    SWOT data supports various applications:
    
    ### Hydrology
    - Monitoring river discharge and lake storage changes
    - Studying floodplain dynamics and seasonal variations
    - Tracking groundwater-surface water interactions
    
    ### Oceanography
    - Studying ocean circulation and mesoscale eddies
    - Monitoring sea level rise and coastal changes
    - Understanding ocean-atmosphere interactions
    
    ### Climate Science
    - Understanding the global water cycle
    - Studying water cycle response to climate change
    - Improving climate models and predictions
    
    ### Water Management
    - Informing sustainable water resource management
    - Supporting drought and flood monitoring
    - Enhancing water security assessments
    
    ## 🔭 Technology
    SWOT uses a Ka-band radar interferometer (KaRIn) to make high-resolution elevation measurements of water surfaces. This technology enables:
    - Measurement of water height with centimeter-level accuracy
    - Two-dimensional mapping of water surfaces
    - Coverage of nearly all rivers wider than 100 meters
    - Monitoring of lakes larger than 6 hectares
    
    ## 📈 Data Products
    - **Level 1**: Calibrated radar measurements
    - **Level 2**: Geolocated water surface elevations
    - **Level 3**: Gridded water surface elevation products
    - **Level 4**: Model-data fusion products
    """)

elif st.session_state.page == 'Contact Us':
    st.title("📧 Contact Us")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ## Get in Touch
        
        We'd love to hear from you! Whether you have questions about the dashboard, 
        need technical support, or want to collaborate on water monitoring research.
        
        ### 📧 Email
        **080mswre006.bishal@pcampus.edu.np**
        
        ### 🏢 Institution
        Pulchowk Campus, Institute of Engineering  
        Tribhuvan University  
        Lalitpur, Nepal
        
        ### 🔬 Research Areas
        - Water Surface Elevation Monitoring
        - SWOT Satellite Data Analysis
        - Nepal Water Resources
        - Remote Sensing Applications
        """)
    
    with col2:
        st.markdown("""
        ## Feedback & Support
        
        Your feedback helps us improve this dashboard. Please let us know:
        
        - 🐛 **Bug Reports**: Found an issue? Let us know!
        - 💡 **Feature Requests**: Ideas for new functionality
        - 📊 **Data Issues**: Problems with data loading or visualization
        - 📝 **Documentation**: Suggestions for better documentation
        
        ## Collaboration Opportunities
        
        Interested in collaborating? We welcome:
        - Research partnerships
        - Data sharing initiatives  
        - Technical contributions
        - Educational outreach
        """)
    
    # Contact form
    st.markdown("### Send us a message:")
    with st.form("contact_form"):
        name = st.text_input("Your Name")
        email = st.text_input("Your Email")
        subject = st.selectbox("Subject", ["General Inquiry", "Technical Support", "Bug Report", "Feature Request", "Collaboration"])
        message = st.text_area("Message", height=100)
        
        if st.form_submit_button("Send Message"):
            if name and email and message:
                st.success("Thank you for your message! We'll get back to you soon.")
            else:
                st.error("Please fill in all required fields.")

# Footer
st.markdown("---")
st.caption("Developed for Nepal Lakes WSE Monitoring using SWOT data 🌊 | Powered by Streamlit, Plotly, Folium, and Geopandas")