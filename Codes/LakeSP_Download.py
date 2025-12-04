# SWOT_LakeSP_Downloader_Auto.py
import earthaccess
import os
import getpass
from datetime import datetime
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import traceback
import pandas as pd
import time  # For optional delay


def load_lake_data(csv_file, sn):
    """Load lake data from CSV and return details for the specified SN by searching column headers"""
    try:
        df = pd.read_csv(csv_file, dtype={'pass_full': str, 'pass_part': str})
        
        # Use the exact column names we know exist
        columns_map = {
            'SN': 'SN',
            'lake_name': 'lake_name', 
            'p_lon': 'p_lon',
            'p_lat': 'p_lat',
            'pass_full': 'pass_full',
            'pass_part': 'pass_part'
        }
        
        # Convert SN values to integers
        df['SN'] = pd.to_numeric(df['SN'], errors='coerce').fillna(0).astype(int)
        
        lake_row = df[df['SN'] == int(sn)]
        if lake_row.empty:
            print(f"❌ SN {sn}: No lake found in CSV.")
            return None
        
        lake_data = lake_row.iloc[0]
        lake_name = lake_data[columns_map['lake_name']]
        p_lon = float(lake_data[columns_map['p_lon']])
        p_lat = float(lake_data[columns_map['p_lat']])
        point = (p_lon, p_lat)

        # Process pass_full
        pass_full = str(lake_data[columns_map['pass_full']]).strip()
        if ';' in pass_full:
            pass_list = [p.strip().zfill(3) for p in pass_full.split(';') if p.strip()]
        else:
            pass_list = [pass_full.zfill(3)]
        
        # Process pass_part (if column exists and has data)
        if 'pass_part' in columns_map:
            pass_part = str(lake_data[columns_map['pass_part']]).strip()
            if pass_part and pass_part != 'nan' and pass_part != '' and pass_part != 'None':
                if ';' in pass_part:
                    pass_list.extend([p.strip().zfill(3) for p in pass_part.split(';') if p.strip()])
                else:
                    pass_list.append(pass_part.zfill(3))
        
        # Remove duplicates and filter out invalid entries
        pass_list = list(set([p for p in pass_list if p.isdigit() and len(p) == 3]))

        print(f"📊 Using pass numbers: {pass_list}")

        return {
            'lake_name': lake_name,
            'point': point,
            'pass_list': pass_list
        }
    except Exception as e:
        print(f"❌ SN {sn}: Error reading CSV: {e}")
        traceback.print_exc()
        return None


def check_netrc_authentication():
    """Check if .netrc file authentication is successful"""
    netrc_path = os.path.join(os.path.expanduser("~"), ".netrc")
    try:
        os.environ['NETRC'] = netrc_path
        earthaccess.search_data(short_name='SWOT_L2_HR_LakeSP_2.0', count=1)
        print("✅ SUCCESS! .netrc authentication successful!")
        return True
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        print(f"Ensure .netrc file exists at {netrc_path} with:")
        print("machine urs.earthdata.nasa.gov")
        print("login your_username")
        print("password your_password")
        return False


def manual_earthdata_login():
    """Fallback to manual authentication if .netrc fails"""
    print("\n🔐 .netrc authentication failed. Falling back to manual login.")
    username = input("Enter NASA Earthdata Username: ").strip()
    password = getpass.getpass("Enter NASA Earthdata Password: ").strip()
    if not username or not password:
        print("❌ Username and password cannot be empty")
        return False
    try:
        os.environ['EARTHDATA_USERNAME'] = username
        os.environ['EARTHDATA_PASSWORD'] = password
        auth = earthaccess.login()
        if auth.authenticated:
            print("✅ SUCCESS! Manual login successful!")
            return True
        else:
            print("❌ Manual login failed - credentials rejected")
            return False
    except Exception as e:
        print(f"❌ Manual login error: {e}")
        return False


def fallback_download(granules, output_dir, max_retries=3, timeout=60):
    """Fallback download function using requests"""
    os.makedirs(output_dir, exist_ok=True)
    downloaded_files = []
    session = requests.Session()
    retries = Retry(total=max_retries, backoff_factor=1, status_forcelist=[502, 503, 504, 401])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    auth = (os.environ.get('EARTHDATA_USERNAME'), os.environ.get('EARTHDATA_PASSWORD'))

    print(f"📥 Downloading {len(granules)} files...")
    
    for i, granule in enumerate(granules, 1):
        try:
            url = granule.data_links()[0]
            filename = os.path.join(output_dir, url.split("/")[-1])
            
            with session.get(url, stream=True, timeout=timeout, auth=auth) as response:
                response.raise_for_status()
                with open(filename, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            downloaded_files.append(filename)
        except Exception as e:
            print(f"❌ Failed to download file {i}: {e}")
            continue
    
    return downloaded_files


def download_swot_lakesp(location, time_scale, pass_list, lake_name, sn):
    """Download SWOT LakeSP data based on criteria"""
    try:
        start_date, end_date = time_scale.split(' to ')
        start_date = datetime.strptime(start_date.strip(), '%Y-%m-%d')
        end_date = datetime.strptime(end_date.strip(), '%Y-%m-%d')

        search_params = {
            'short_name': 'SWOT_L2_HR_LakeSP_2.0',
            'temporal': (start_date, end_date),
            'point': location
        }

        print(f"\n🔍 Searching for SWOT LakeSP data for {lake_name}...")
        results = earthaccess.search_data(**search_params)

        if not results:
            print(f"❌ {lake_name}: No data found in time/location range.")
            return False

        # Filter for Prior + pass
        filtered_results = [
            r for r in results
            if ("_Prior_" in r['umm']['GranuleUR']) and
               any(f"_{p}_AS_" in r['umm']['GranuleUR'] for p in pass_list)
        ]

        if not filtered_results:
            print(f"❌ {lake_name}: No granules matched '_Prior_' and pass filter.")
            return False

        print(f"✅ Found {len(filtered_results)} matching granules.")

        # Output folder
        output_dir = os.path.join("swot_lakesp_data", lake_name)
        os.makedirs(output_dir, exist_ok=True)

        # Authenticate
        netrc_path = os.path.join(os.path.expanduser("~"), ".netrc")
        os.environ['NETRC'] = netrc_path
        earthaccess.login(strategy="netrc")

        print(f"📥 Downloading {len(filtered_results)} files...")
        try:
            # Remove verbose parameter to avoid the error
            files = earthaccess.download(filtered_results, output_dir)
            print(f"✅ Successfully downloaded {len(files)} files to {output_dir}")
            return True
        except Exception as e:
            # Don't show the specific error message for earthaccess
            print("🔁 Trying fallback download method...")
            files = fallback_download(filtered_results, output_dir)
            if files:
                print(f"✅ Successfully downloaded {len(files)} files to {output_dir}")
                return True
            else:
                print(f"❌ Failed to download any files for {lake_name}")
                return False

    except Exception as e:
        print(f"❌ Error during download for {lake_name}: {e}")
        return False


def main():
    """Main function: automate download for SN 1 to 44"""
    print("🌍 NASA Earthdata SWOT LakeSP Prior Downloader (Auto: SN 1–44)")
    print("=" * 60)

    # Authentication
    if not check_netrc_authentication():
        if not manual_earthdata_login():
            print("❌ Authentication failed. Exiting.")
            return

    # CSV file path
    csv_file = "D:\\SWOT_Mission_Data_Download\\NP_Lake_Datasets.csv"
    time_scale = "2022-12-16 to 2025-08-10"

    # Check CSV exists
    if not os.path.exists(csv_file):
        print(f"❌ CSV file not found: {csv_file}")
        return

    # Track summary
    success_count = 0
    fail_count = 0

    # Loop over SN 1 to 44
    for sn in range(1, 44):
        print(f"\n📌 --- Processing SN {sn} ---")

        lake_data = load_lake_data(csv_file, sn)
        if not lake_data:
            print(f"🛑 Skipping SN {sn}.")
            fail_count += 1
            continue

        success = download_swot_lakesp(
            location=lake_data['point'],
            time_scale=time_scale,
            pass_list=lake_data['pass_list'],
            lake_name=lake_data['lake_name'],
            sn=sn
        )

        if success:
            print(f"🎉 SUCCESS: {lake_data['lake_name']} (SN {sn})")
            success_count += 1
        else:
            print(f"❌ FAILED: {lake_data['lake_name']} (SN {sn})")
            fail_count += 1

        # Optional: delay to avoid overwhelming server
        time.sleep(2)

    # Final Summary
    print("\n" + "=" * 60)
    print(f"✅ COMPLETED: {success_count} Success, {fail_count} Failed")
    print("📁 All data saved in ./swot_lakesp_data/")


if __name__ == "__main__":
    main()