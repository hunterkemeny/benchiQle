import os
import requests
import json
from datetime import datetime

def get_qiskit_releases_data(package_name: str) -> dict:
    response = requests.get(f"https://pypi.org/pypi/{package_name}/json")
    if response.status_code == 200:
        data = response.json()
        return data["releases"].items()
    return None

def get_qiskit_versions_info() -> []:
    data_items = get_qiskit_releases_data("qiskit")

    # Filter releases starting from 2023-08
    # Starting with qiskit 0.45, qiskit and qiskit-terra will have the same version
    return filter_by_date(data_items,2023,8)

def get_qiskit_terra_versions_info() -> []:
    data_items = get_qiskit_releases_data("qiskit-terra")

    # Filter releases starting from 2020-03 (qiskit-terra version 0.13.0)
    return filter_by_date(data_items,2020,3)

def get_qiskit_versions_list() -> []:
    qiskit_versions_info = get_qiskit_versions_info()
    versions_only = []
    for item in qiskit_versions_info:
        for key, value in item.items():
            if key == "version":
                versions_only.append(value)
    return versions_only

def find_latest_version(versions: []) -> str:
  if not versions:
    return ""
  
  # Split each version string into a tuple of integers
  version_tuples = [tuple(map(int, v.split("."))) for v in versions]

  # Sort
  sorted_versions = sorted(version_tuples, reverse=True)

  # Convert the latest version tuple back to string
  latest_version = ".".join(map(str, sorted_versions[0]))
  return latest_version

def compare_versions(version_1:str, version_2: str) -> str:
    # Split version strings into lists of ints
    v1 = list(map(int, version_1.split(".")))
    v2 = list(map(int, version_2.split(".")))

    # Compare
    for i in range(max(len(v1), len(v2))):
        n1 = v1[i] if i < len(v1) else 0
        n2 = v2[i] if i < len(v2) else 0
        if n1 < n2:
            return version_2
        return version_1

def same_minor(version_1: str, version_2: str) -> bool:
    # Split version strings into lists of ints
    v1 = list(map(int, version_1.split(".")))
    v2 = list(map(int, version_2.split(".")))

    return v1[:2] == v2[:2]

def filter_by_date(data_items: dict, y: int, m: int) -> []:
    # Temporary control dictionary for package release info for version, date and python version
    temp = {}

    for release, release_info in data_items:
        # Skip RCs
        if "rc" in release:
            continue

        date_str = release_info[0]["upload_time"]
        dt = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S")
        year = dt.year
        month = dt.month
        
        if (year == y and month < m) or year < y:
            continue

        python_version = release_info[0]["requires_python"]

        # Parse the release string of format "x.y.z" into a list of "x","y","z"
        major_minor_patch_list = release.split(".")
        major_minor = ".".join(major_minor_patch_list[:2])

        # Get latest patch
        patch_number = int(major_minor_patch_list[2])
        temp_info = temp.get(major_minor)
        previous_patch_number = -1 if not temp_info else temp_info[0]

        if previous_patch_number < patch_number:
            # Replace to latest patch version found
            temp[major_minor] = (patch_number, {"version":release, "date": dt.strftime("%Y-%m-%d"), "python_version": python_version})
            continue

    filtered_releases = []
    for _, value in temp.items():
        filtered_releases.append(value[1])

    return filtered_releases

def write_versions_to_file(versions: [], filename: str):
    file_path = os.path.abspath(os.path.join( os.path.dirname( __file__ ), "benchmarking", filename))
    with open(file_path,"w") as file:
        json.dump(versions, file, indent=4)

def get_version_date(package_name: str, input_version:str) -> str:
    data_items = data_items = get_qiskit_releases_data(package_name)
    for release, release_info in data_items:
        if release == input_version:
            # Remove time from date format "%Y-%m-%dT%H:%M:%S"
            date_time= release_info[0]["upload_time"]
            return date_time.split('T', 1)[0]
    return "Invalid version"

qiskit_info = get_qiskit_versions_info()
write_versions_to_file(qiskit_info, "qiskit.json")
print("qiskit versions:", sep='\n')
print(*qiskit_info, sep='\n')

#NOTE: qiskit-terra package used up until Aug 2023. Future experiment runs to use qiskit package only.
qiskit_terra_info = get_qiskit_terra_versions_info()
write_versions_to_file(qiskit_terra_info, "qiskit_versions.json")
print("qiskit-terra versions:", sep='\n')
print(*qiskit_terra_info, sep='\n')