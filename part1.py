# %% read csv file
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

#%%
df = pd.read_csv("2012.csv")


# %% show stats
df.describe(include="all").to_csv("stats.csv")


# %% make sure these number columns contain numbers only
# if a value cannot be converted to a number, make it NaN
from tqdm import tqdm

cols = [
    "perobs",
    "perstop",
    "age",
    "weight",
    "ht_feet",
    "ht_inch",
    "datestop",
    "timestop",
    "xcoord",
    "ycoord",
]
for col in tqdm(cols, desc="convert to number"):
    df[col] = pd.to_numeric(df[col], errors="coerce")


# %% drop rows with invalid numberic value
df = df.dropna()


# %% make datetime column
df["datestop"] = df["datestop"].astype(str).str.zfill(8)
df["timestop"] = df["timestop"].astype(str).str.zfill(4)


#%%
from datetime import datetime


def make_datetime(datestop, timestop):
    year = int(datestop[-4:])
    month = int(datestop[:2])
    day = int(datestop[2:4])

    hour = int(timestop[:2])
    minute = int(timestop[2:])

    return datetime(year, month, day, hour, minute)


df["datetime"] = df.apply(
    lambda row: make_datetime(row["datestop"], row["timestop"]),
    axis=1,
)


# %% make height column
df["height"] = (df["ht_feet"] * 12 + df["ht_inch"]) * 2.54


# %% make lat/lon columns
import pyproj

srs = (
    "+proj=lcc +lat_1=41.03333333333333 "
    "+lat_2=40.66666666666666 +lat_0=40.16666666666666 +lon_0=-74 "
    "+x_0=300000.0000000001 +y_0=0 "
    "+ellps=GRS80 +datum=NAD83 +to_meter=0.3048006096012192 +no_defs"
)

p = pyproj.Proj(srs)

coords = df.apply(lambda r: p(r["xcoord"], r["ycoord"], inverse=True), axis=1)
df["lat"] = [c[1] for c in coords]
df["lon"] = [c[0] for c in coords]


# %% read the spec file and replace values in df with the matching labels
import numpy as np

value_labels = pd.read_excel(
    "2012 SQF File Spec.xlsx", sheet_name="Value Labels", skiprows=range(4)
)
value_labels["Field Name"] = value_labels["Field Name"].fillna(method="ffill")
value_labels["Field Name"] = value_labels["Field Name"].str.lower()
value_labels["Value"] = value_labels["Value"].fillna(" ")
vl_mapping = value_labels.groupby("Field Name").apply(
    lambda x: dict([(row["Value"], row["Label"]) for row in x.to_dict("records")])
)

cols = [col for col in df.columns if col in vl_mapping]

for col in tqdm(cols):
    df[col] = df[col].apply(lambda val: vl_mapping[col].get(val, np.nan))


# %% plot height
import seaborn as sns

sns.histplot(data=df,x="height",color="purple")

#  %%
sns.displot(df["weight"])

# %%
sns.displot(df["age"])

# %% remove rows with invalid age/weight
df = df[(df["age"] <= 100) & (df["age"] >= 10)]
df = df[(df["weight"] <= 350) & (df["weight"] >= 50)]


# %% plot month
sns.countplot(x=df["datetime"].dt.month)


# %% plot day of week
ax = sns.countplot(x=df["datetime"].dt.weekday)
ax.set_xticklabels(["Mon", "Tue", "Wed", "Thur", "Fri", "Sat", "Sun"])
ax.set(xlabel="day of week", title="# of incidents by day of weeks")
ax.get_figure().savefig("test.png")



# %% plot hour
ax = sns.countplot(x=df["datetime"].dt.hour)
ax.get_figure().savefig("break-down-by-hour.png")


# %% plot xcoord / ycoord
sns.scatterplot(data=df[:100], x="xcoord", y="ycoord")


# %%
import folium

m = folium.Map((51.0447, -114.0719))
m


# %% plot lat / lon of murder cases on an actual map
import folium

m = folium.Map((40.7128, -74.0060))

for r in df[["lat", "lon"]][df["detailcm"] == "MURDER"].to_dict("records"):
    folium.CircleMarker(location=(r["lat"], r["lon"]), radius=1).add_to(m)

m

# %% plot lat / lon of terrorism cases on an actual map

m = folium.Map((40.7128, -74.0060))

for r in df[["lat", "lon"]][df["detailcm"] == "TERRORISM"].to_dict("records"):
    folium.CircleMarker(location=(r["lat"], r["lon"]), radius=1).add_to(m)

m

# %% plot race
ax = sns.countplot(data=df, x="race")
for p in ax.patches:
   ax.annotate('{:.1f}'.format(p.get_height()), (p.get_x()+0.25, p.get_height()+0.01))
ax.set_xticklabels(rotation=0)
plt.show()

#%% arrestmade
ax = sns.countplot(data=df, x="arstmade")
for p in ax.patches:
   ax.annotate('{:.1f}'.format(p.get_height()), (p.get_x()+0.25, p.get_height()+0.01))
ax.set_xticklabels(rotation=0)
plt.show()



# %% plot race wrt city
sns.countplot(data=df, y="race", hue="city")



# %% plot top crimes where physical forces used
pf_used = df[
    (df["pf_hands"] == "YES")
    | (df["pf_wall"] == "YES")
    | (df["pf_grnd"] == "YES")
    | (df["pf_drwep"] == "YES")
    | (df["pf_ptwep"] == "YES")
    | (df["pf_baton"] == "YES")
    | (df["pf_hcuff"] == "YES")
    | (df["pf_pepsp"] == "YES")
    | (df["pf_other"] == "YES")
]

sns.countplot(
    data=pf_used,
    y="detailcm",
    order=pf_used["detailcm"].value_counts(ascending=False).keys()[:10],
)


# %% plot percentage of each physical forces used
pfs = [col for col in df.columns if col.startswith("pf_")]
pf_counts = (df[pfs] == "YES").sum()
sns.barplot(y=pf_counts.index, x=pf_counts.values / df.shape[0])


# %% drop columns that are not used in analysis
df = df.drop(
    columns=[
        # processed columns
        "datestop",
        "timestop",
        "ht_feet",
        "ht_inch",
        "xcoord",
        "ycoord",
        # not useful
        "year",
        "recstat",
        "crimsusp",
        "dob",
        "ser_num",
        "arstoffn",
        "sumoffen",
        "compyear",
        "comppct",
        "othfeatr",
        "adtlrept",
        "dettypcm",
        "linecm",
        "repcmd",
        "revcmd",
        # location of stop
        # only use coord and city
        "addrtyp",
        "rescode",
        "premtype",
        "premname",
        "addrnum",
        "stname",
        "stinter",
        "crossst",
        "aptnum",
        "state",
        "zip",
        "addrpct",
        "sector",
        "beat",
        "post",
    ]
)

# %% modify one column
df["trhsloc"] = df["trhsloc"].fillna("NEITHER")


#%%
df[[   
    "age", "weight", "race", "lat", "lon", 
    "pct", "arstmade", "frisked", "city"
    
    ]].describe(include="all")

# %% remove all rows with NaN
df = df.dropna()


# %% save dataframe to a file
df.to_pickle("data.pkl")

# %%
reason_used = [col for col in df.columns if col.startswith("cs_") or col.startswith("rf_")]
(df[reason_used] == "YES").sum(axis=1)

# %%
df["number_of_reasons"] = (df[reason_used] == "YES").sum(axis=1)
sns.countplot(
    data=df,
    y="forceuse", 
    hue="number_of_reasons"
)
# %%
sns.countplot(data=df, y="number_of_reasons")

# %%
df = pd.read_csv("New York 2012 race population estimate.csv")

# %%
import matplotlib.pyplot as plt
#2012 population data by race
#x = ['Hispanic', 'White', 'Black', 'Asian', 'Other or Multiple Race']

# %%
df = df.drop(
    columns=["Unnamed: 3"]
)
#%%
ax = df.plot.bar(x='Race', stacked=True, color=['red','blue'])
ax.set_title('2012 New York Population Estimates by Race/Ethnicity(in millions)')
ax.set_ylim(0,3000000)
ax.set_xticklabels(['Hispanic','White','Black', 'Asian', 'Others'], rotation=0)
""" 
df.plot.bar(
    x="Race", stacked=True, title='2012 New York Population Estimates by Race/Ethnicity'
    ) """
# %%
