#!/usr/bin/env python3
"""
Test a SINGLE molecule to see what's going wrong with xTB
"""

import subprocess
import tempfile
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem

# Simple test molecule: benzene
SMILES = "c1ccccc1"

print("🧪 Testing xTB with benzene...")
print(f"SMILES: {SMILES}")

# Step 1: Check if xTB is installed
print("\n1️⃣ Checking xTB installation...")
try:
    result = subprocess.run(["xtb", "--version"], capture_output=True, text=True)
    print(f"✓ xTB found: {result.stdout.strip()}")
except FileNotFoundError:
    print("❌ xTB not found! Install with: conda install -c conda-forge xtb")
    exit(1)

# Step 2: Generate 3D structure
print("\n2️⃣ Generating 3D structure...")
mol = Chem.MolFromSmiles(SMILES)
mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())

try:
    AllChem.UFFOptimizeMolecule(mol, maxIters=50)
    print("✓ UFF optimization succeeded")
except:
    print("⚠️  UFF failed, continuing anyway...")

# Step 3: Write SDF
print("\n3️⃣ Writing SDF file...")
with tempfile.TemporaryDirectory() as tmpdir:
    tmpdir = Path(tmpdir)
    sdf_path = tmpdir / "test.sdf"
    
    w = Chem.SDWriter(str(sdf_path))
    w.write(mol)
    w.close()

    # Step 4: Run xTB GFN2
    print("\n4️⃣ Running xTB GFN2...")
    cmd = ["xtb", str(sdf_path), "--gfn", "2", "--sp"]
    
    try:
        result = subprocess.run(
            cmd, 
            cwd=tmpdir, 
            capture_output=True, 
            text=True,
            timeout=30
        )
        
        print(f"\n📜 STDOUT:\n{result.stdout[:500]}...") # First 500 chars
        print(f"\n⚠️  STDERR:\n{result.stderr}")
        
        if result.returncode == 0:
            print("\n✅ SUCCESS! xTB GFN2 is working.")
        else:
            print(f"\n❌ FAILURE! Return code: {result.returncode}")
            
    except Exception as e:
        print(f"\n❌ EXCEPTION: {e}")