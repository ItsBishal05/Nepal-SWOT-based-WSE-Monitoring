# SWOT_Node_Downloader_Organized_KeepTemp.py
import earthaccess
import os
import getpass
from datetime import datetime
import pandas as pd
import time
import logging
from contextlib import redirect_stdout, redirect_stderr
import io
import shutil
import glob
import re
import traceback


def load_orbit_data(csv_file):
    """Load orbit IDs from CSV file"""
    try:
        df = pd.read_csv(csv_file)
        
        # Look for orbit column (case insensitive)
        orbit_columns = [col for col in df.columns if 'orbit' in col.lower()]
        
        if not orbit_columns:
            print("❌ No orbit column found in CSV")
            return []
        
        orbit_column = orbit_columns[0]
        orbit_ids = df[orbit_column].dropna().astype(str).tolist()
        
        # Clean and format orbit IDs
        formatted_orbits = []
        for orbit in orbit_ids:
            # Remove any non-digit characters and ensure proper formatting
            orbit_clean = re.sub(r'\D', '', orbit)
            if orbit_clean:
                # Pad with leading zeros to make 3-digit orbit numbers
                orbit_clean = orbit_clean.zfill(3)
                formatted_orbits.append(orbit_clean)
        
        # Remove duplicates
        unique_orbits = list(set(formatted_orbits))
        print(f"✅ Loaded {len(unique_orbits)} unique orbit IDs from CSV")
        return unique_orbits
        
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        traceback.print_exc()
        return []


def download_orbit_data(orbit_ids, time_scale, base_download_dir):
    """Download SWOT node data for given orbit IDs with precise filtering"""
    try:
        start_date, end_date = time_scale.split(' to ')
        start_date = datetime.strptime(start_date.strip(), '%Y-%m-%d')
        end_date = datetime.strptime(end_date.strip(), '%Y-%m-%d')

        print(f"🔍 Searching for SWOT node data for {len(orbit_ids)} orbits...")
        print(f"   Orbit list: {', '.join(orbit_ids)}")

        all_results = []

        # Bounding box for Nepal (lon_min, lat_min, lon_max, lat_max)
        bbox_nepal = (80.0, 26.0, 88.0, 30.5)  # Nepal bounding box

        for i, orbit_str in enumerate(orbit_ids, 1):
            print(f"   ➤ Searching orbit {orbit_str} ({i}/{len(orbit_ids)})...")

            # Use more specific search parameters
            search_params = {
                'short_name': 'SWOT_L2_HR_RiverSP_D',
                'temporal': (start_date, end_date),
                'bounding_box': bbox_nepal,
                'cloud_hosted': True
            }

            try:
                # Search for granules
                results = earthaccess.search_data(**search_params)
                
                if results:
                    print(f"     ✅ Found {len(results)} total granules in Nepal region")
                    
                    # STRICT FILTERING: Match exact orbit and node data pattern
                    orbit_results = []
                    for granule in results:
                        granule_name = granule['umm']['GranuleUR']
                        
                        # CRITICAL FILTERS:
                        # 1. Must contain exact orbit number (case insensitive)
                        # 2. Must contain "_node_" (case insensitive)
                        # 3. Must contain orbit ID and AS pattern together: *_OrbitID_AS_*
                        granule_name_lower = granule_name.lower()
                        
                        # COMBINED FILTER: Look for pattern *_orbit_AS_* in the same segment
                        if (f"_{orbit_str}_as_" in granule_name_lower and 
                            "_node_" in granule_name_lower and
                            "swot_l2_hr_riversp" in granule_name_lower):
                            
                            orbit_results.append(granule)
                    
                    if orbit_results:
                        print(f"       ✅ Found {len(orbit_results)} node granules for orbit {orbit_str} with _{orbit_str}_AS_ pattern")
                        all_results.extend(orbit_results)
                        
                        # Show sample granule names
                        if orbit_results:
                            sample_name = orbit_results[0]['umm']['GranuleUR']
                            print(f"       📋 Sample: {sample_name}")
                    else:
                        print(f"       ⚠️ No node granules found for orbit {orbit_str} with _{orbit_str}_AS_ pattern")
                else:
                    print(f"     ⚠️ No granules found in Nepal region for orbit {orbit_str}")
                    
            except Exception as e:
                print(f"     ❌ Error searching orbit {orbit_str}: {e}")
                continue

            time.sleep(1)  # Be kind to CMR - increased delay

        if not all_results:
            print("❌ No node data found for any of the specified orbits after filtering")
            return False

        total_files = len(all_results)
        print(f"\n✅ Found {total_files} filtered '_node_' files with '_OrbitID_AS_' pattern across {len(orbit_ids)} orbits")
        print(f"📥 Downloading all files to organized directory structure...")

        # Create base download directory
        os.makedirs(base_download_dir, exist_ok=True)

        # Download files to temporary directory
        temp_download_dir = os.path.join(base_download_dir, "temp_downloads")
        os.makedirs(temp_download_dir, exist_ok=True)

        # Download in smaller batches to avoid timeouts
        batch_size = 5
        success_count = 0
        
        for i in range(0, len(all_results), batch_size):
            batch = all_results[i:i + batch_size]
            print(f"   Downloading batch {i//batch_size + 1}/{(len(all_results)-1)//batch_size + 1} ({len(batch)} files)...")
            
            # Suppress verbose logging during download
            logging.getLogger("earthaccess").setLevel(logging.CRITICAL)
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                try:
                    files = earthaccess.download(batch, temp_download_dir)
                    if files:
                        success_count += len(files)
                        print(f"     ✅ Downloaded {len(files)} files")
                    else:
                        print(f"     ⚠️ Batch download failed")
                except Exception as e:
                    print(f"     ❌ Download error: {e}")
            
            logging.getLogger("earthaccess").setLevel(logging.INFO)
            time.sleep(2)  # Delay between batches

        print(f"✅ Downloaded {success_count}/{total_files} files")
        
        # Organize files into orbit folders
        organized_count = organize_files_by_orbit(temp_download_dir, base_download_dir, orbit_ids)
        
        # KEEP TEMPORARY FOLDER - Do not delete it
        print(f"📁 Temporary download directory kept: {temp_download_dir}")
        
        return organized_count > 0

    except Exception as e:
        print(f"❌ Error during download: {e}")
        traceback.print_exc()
        return False


def organize_files_by_orbit(temp_download_dir, base_download_dir, orbit_ids):
    """Organize downloaded files into orbit-specific folders"""
    print(f"\n📁 Organizing files into orbit folders from: {temp_download_dir}")
    
    # Get all files in the temporary directory (including subdirectories)
    all_files = []
    for root, dirs, files in os.walk(temp_download_dir):
        for file in files:
            file_path = os.path.join(root, file)
            all_files.append(file_path)
    
    print(f"   Found {len(all_files)} total files in temporary directory")
    
    if not all_files:
        print("❌ No files found in temporary directory")
        return 0
    
    # Debug: Show what files we found
    print("\n📋 Files found in temp directory:")
    for i, file_path in enumerate(all_files[:10]):  # Show first 10 files
        print(f"   {i+1:2d}. {os.path.basename(file_path)}")
    if len(all_files) > 10:
        print(f"   ... and {len(all_files) - 10} more files")
    
    organized_count = 0
    
    for orbit in orbit_ids:
        orbit_folder = os.path.join(base_download_dir, f"orbit_{orbit}")
        os.makedirs(orbit_folder, exist_ok=True)
        
        orbit_files = []
        for file_path in all_files:
            filename = os.path.basename(file_path).lower()
            # COMBINED FILTER: Look for pattern *_orbit_AS_* in the same segment
            if (f"_{orbit}_as_" in filename and 
                "_node_" in filename):
                orbit_files.append(file_path)
        
        if orbit_files:
            print(f"\n   ➤ Orbit {orbit}: {len(orbit_files)} files with _{orbit}_AS_ pattern")
            for file_path in orbit_files:
                dest_path = os.path.join(orbit_folder, os.path.basename(file_path))
                if not os.path.exists(dest_path):  # Avoid overwriting
                    shutil.move(file_path, dest_path)
                    organized_count += 1
                    print(f"      ✓ Moved: {os.path.basename(file_path)}")
        else:
            print(f"   ➤ Orbit {orbit}: No matching files found with _{orbit}_AS_ pattern")
    
    print(f"\n✅ Organized {organized_count} files into orbit folders")
    
    # Check what files remain in temp directory
    remaining_files = []
    for root, dirs, files in os.walk(temp_download_dir):
        for file in files:
            remaining_files.append(os.path.join(root, file))
    
    if remaining_files:
        print(f"📊 {len(remaining_files)} files remain in temp directory")
        print("   These files may not match the _OrbitID_AS_ pattern or orbit criteria")
        print("   They will be kept for reference")
    
    return organized_count


def check_existing_downloads(base_download_dir, orbit_ids):
    """Check if files already exist for these orbits to avoid re-downloading"""
    orbits_to_download = orbit_ids.copy()
    
    for orbit in orbit_ids:
        orbit_folder = os.path.join(base_download_dir, f"orbit_{orbit}")
        if os.path.exists(orbit_folder):
            # Check if folder has any files
            orbit_files = []
            for root, dirs, files in os.walk(orbit_folder):
                for file in files:
                    orbit_files.append(os.path.join(root, file))
            
            if orbit_files:
                if orbit in orbits_to_download:
                    orbits_to_download.remove(orbit)
                    print(f"ℹ️ Files for orbit {orbit} already exist ({len(orbit_files)} files), skipping")
    
    return orbits_to_download


def manual_earthdata_login():
    """Manual authentication"""
    print("\n🔐 Manual Earthdata login required")
    username = input("Enter NASA Earthdata Username: ").strip()
    password = getpass.getpass("Enter NASA Earthdata Password: ").strip()
    
    if not username or not password:
        print("❌ Username and password cannot be empty")
        return False
    
    try:
        auth = earthaccess.login(strategy="interactive", persist=True)
        if auth:
            print("✅ Manual login successful")
            return True
        else:
            print("❌ Manual login failed")
            return False
    except Exception as e:
        print(f"❌ Manual login error: {e}")
        return False


def main():
    """Main function: Download SWOT node data based on orbit IDs from CSV"""
    print("🌍 NASA Earthdata SWOT Node Data Downloader (Organized)")
    print("=" * 70)
    print("📋 Strategy: Organized by orbit folders + Nepal bounding box")
    print("🔍 Filter: *_OrbitID_AS_* pattern (combined filter)")
    print("📁 Temporary folder will be kept after completion")
    print("=" * 70)

    # Authentication
    print("🔐 Attempting authentication...")
    
    netrc_path = os.path.join(os.path.expanduser("~"), ".netrc")
    if os.path.exists(netrc_path):
        try:
            os.environ['NETRC'] = netrc_path
            auth = earthaccess.login(strategy="netrc")
            if auth:
                print("✅ Authenticated using .netrc file")
            else:
                if not manual_earthdata_login():
                    print("❌ Authentication failed. Exiting.")
                    return
        except Exception as e:
            print(f"❌ .netrc authentication error: {e}")
            if not manual_earthdata_login():
                print("❌ Authentication failed. Exiting.")
                return
    else:
        if not manual_earthdata_login():
            print("❌ Authentication failed. Exiting.")
            return

    # Configuration
    csv_file = "D:\\SWOT_Mission_Data_Download\\Orbit_ID.csv"  # Update with your CSV path
    time_scale = "2022-04-01 to 2025-08-10"
    base_download_dir = "swot_node_data"

    # Check CSV exists
    if not os.path.exists(csv_file):
        print(f"❌ CSV file not found: {csv_file}")
        print("Please update the 'csv_file' variable with your CSV file path")
        return

    # Load orbit IDs from CSV
    print("📊 Loading orbit IDs from CSV...")
    orbit_ids = load_orbit_data(csv_file)

    if not orbit_ids:
        print("❌ No valid orbit IDs found in CSV")
        return

    # Check for existing downloads
    orbits_to_process = check_existing_downloads(base_download_dir, orbit_ids)
    
    if not orbits_to_process:
        print("✅ All orbit data already downloaded!")
        return

    # Download node data
    print(f"\n📥 Processing {len(orbits_to_process)} orbits that need downloading...")
    success = download_orbit_data(orbits_to_process, time_scale, base_download_dir)

    # Final Summary
    print("\n" + "=" * 70)
    if success:
        # Count files in each orbit folder
        total_files = 0
        orbit_counts = {}
        
        for orbit in orbit_ids:
            orbit_folder = os.path.join(base_download_dir, f"orbit_{orbit}")
            if os.path.exists(orbit_folder):
                orbit_files = []
                for root, dirs, files in os.walk(orbit_folder):
                    for file in files:
                        orbit_files.append(os.path.join(root, file))
                file_count = len(orbit_files)
                orbit_counts[orbit] = file_count
                total_files += file_count
        
        print("🎉 DOWNLOAD AND ORGANIZATION COMPLETE")
        print("=" * 70)
        print(f"📁 Base directory: ./{base_download_dir}/")
        print(f"📁 Temporary directory kept: ./{base_download_dir}/temp_downloads/")
        print(f"📊 Total files organized: {total_files}")
        print(f"📊 Orbits with data: {len([o for o in orbit_counts if orbit_counts[o] > 0])}")
        
        # Show detailed orbit-wise breakdown
        print(f"\n📋 Orbit-wise file count:")
        for orbit, count in sorted(orbit_counts.items()):
            status = "✅" if count > 0 else "❌"
            print(f"   {status} Orbit {orbit}: {count} files")
        
        # Show folder structure
        print(f"\n📂 Folder structure:")
        print(f"   {base_download_dir}/")
        print(f"   ├── temp_downloads/  [temporary files - kept for reference]")
        for orbit in sorted(orbit_counts.keys()):
            if orbit_counts[orbit] > 0:
                print(f"   ├── orbit_{orbit}/")
                print(f"   │   └── [ {orbit_counts[orbit]} node data files with _{orbit}_AS_ pattern ]")
            
    else:
        print("❌ DOWNLOAD FAILED OR PARTIAL")
        print("Please check the errors above and try again")
        print(f"Temporary download directory kept for debugging: {base_download_dir}/temp_downloads/")
    
    print("=" * 70)


if __name__ == "__main__":
    # Set up logging to suppress most messages
    logging.basicConfig(level=logging.CRITICAL)
    main()