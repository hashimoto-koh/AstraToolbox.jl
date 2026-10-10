import json, os, re

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.normpath(os.path.join(script_dir, ".."))
demos_dir = os.path.join(project_root, "demos")
tutorials_dir = os.path.join(script_dir, "src", "tutorials")
assets_dir = os.path.join(script_dir, "src", "assets", "tutorials")

os.makedirs(tutorials_dir, exist_ok=True)
os.makedirs(assets_dir, exist_ok=True)

notebook_configs = [
    {
        'nb': os.path.join(demos_dir, '01_2D_parallel_basic.ipynb'),
        'out': os.path.join(tutorials_dir, '01_2d_parallel_basic.md'),
        'images': {
            5: ('Shepp-Logan Phantom', '../assets/tutorials/01_phantom.png'),
            9: ('Sinogram (Forward Projection)', '../assets/tutorials/01_sinogram.png'),
            11: ('Unfiltered Backprojection', '../assets/tutorials/01_unfiltered_bp.png'),
            13: ('FBP Reconstruction', '../assets/tutorials/01_fbp.png'),
            17: ('Reconstruction Method Comparison', '../assets/tutorials/01_comparison.png'),
            19: ('Horizontal Profile Comparison', '../assets/tutorials/01_profile.png'),
        }
    },
    {
        'nb': os.path.join(demos_dir, '02_projector_and_adjoint.ipynb'),
        'out': os.path.join(tutorials_dir, '02_projector_and_adjoint.md'),
        'images': {
            13: ('Linear Operator Transformations', '../assets/tutorials/02_operator.png'),
        }
    },
    {
        'nb': os.path.join(demos_dir, '03_custom_iterative_reconstruction.ipynb'),
        'out': os.path.join(tutorials_dir, '03_custom_iterative_reconstruction.md'),
        'images': {
            5: ('Ground Truth and Noisy Sinogram', '../assets/tutorials/03_noisy_data.png'),
            13: ('Convergence History', '../assets/tutorials/03_convergence.png'),
            15: ('Reconstruction Visual Comparison', '../assets/tutorials/03_recon_comparison.png'),
        }
    },
    {
        'nb': os.path.join(demos_dir, '04_3D_conebeam_FDK.ipynb'),
        'out': os.path.join(tutorials_dir, '04_3d_conebeam_fdk.md'),
        'images': {
            9: ('Cone-Beam Projections', '../assets/tutorials/04_projections.png'),
            13: ('FDK Orthogonal Cross-Sections', '../assets/tutorials/04_fdk_orthogonal.png'),
        }
    },
    {
        'nb': os.path.join(demos_dir, '05_3D_geometry_and_cone_vec.ipynb'),
        'out': os.path.join(tutorials_dir, '05_3d_geometry_and_cone_vec.md'),
        'images': {
            7: ('3D Scanning Trajectory', '../assets/tutorials/05_trajectory.png'),
            11: ('Alignment Calibration Comparison', '../assets/tutorials/05_calibration.png'),
        }
    },
    {
        'nb': os.path.join(demos_dir, '06_conebeam_laminography.ipynb'),
        'out': os.path.join(tutorials_dir, '06_conebeam_laminography.md'),
        'images': {
            7: ('Laminography Trajectories at 0°, 30°, 60° Tilt', '../assets/tutorials/06_trajectories.png'),
            13: ('PCB Reconstructed Layers with SIRT3D', '../assets/tutorials/06_layers.png'),
        }
    },
    {
        'nb': os.path.join(demos_dir, '07_sampling_and_accuracy.ipynb'),
        'out': os.path.join(tutorials_dir, '07_sampling_and_accuracy.md'),
        'images': {
            5: ('High-Contrast Test Phantom', '../assets/tutorials/07_phantom.png'),
            9: ('Relative Error vs Supersampling Rate', '../assets/tutorials/07_error_curve.png'),
            11: ('Anti-Aliasing Difference Image', '../assets/tutorials/07_antialiasing.png'),
        }
    },
    {
        'nb': os.path.join(demos_dir, '08_limited_angle_and_sparse_view.ipynb'),
        'out': os.path.join(tutorials_dir, '08_limited_angle_and_sparse_view.md'),
        'images': {
            7: ('FBP Breakdown under Incomplete Data', '../assets/tutorials/08_fbp_breakdown.png'),
            9: ('SIRT Artifact Suppression', '../assets/tutorials/08_sirt_improvement.png'),
            11: ('DART Discrete Recovery from 15 Projections', '../assets/tutorials/08_dart_recovery.png'),
        }
    },
    {
        'nb': os.path.join(demos_dir, '09_large_scale_and_gpu.ipynb'),
        'out': os.path.join(tutorials_dir, '09_large_scale_and_gpu.md'),
        'images': {
            7: ('3D GPU Processing Throughput', '../assets/tutorials/09_throughput.png'),
            9: ('Tiled Composite Forward Projection', '../assets/tutorials/09_composite.png'),
        }
    },
]

for cfg in notebook_configs:
    if not os.path.exists(cfg['nb']):
        continue
    with open(cfg['nb'], 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    md_content = []
    first_h1 = True
    for i, cell in enumerate(nb['cells']):
        cell_type = cell['cell_type']
        source = ''.join(cell['source']).rstrip()
        
        if cell_type == 'markdown':
            lines = source.splitlines()
            if first_h1 and lines and lines[0].startswith('#'):
                m = re.match(r'^#\s*0?(\d+)\.\s*(.*)$', lines[0])
                if m:
                    lines[0] = f'# Tutorial {int(m.group(1))}: {m.group(2)}'
                first_h1 = False
                source = '\n'.join(lines)
            md_content.append(source)
            md_content.append('')
        elif cell_type == 'code':
            if source.strip():
                md_content.append('```julia')
                md_content.append(source)
                md_content.append('```')
                md_content.append('')
            
            if i in cfg['images']:
                caption, img_path = cfg['images'][i]
                md_content.append(f'![{caption}]({img_path})')
                md_content.append('')
                
    with open(cfg['out'], 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_content))
    print(f"Generated {cfg['out']}")

