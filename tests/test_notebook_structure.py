import json
from pathlib import Path


def test_best_ft_cnn_notebook_contains_gradcam_workflow():
    notebook_path = Path('code/best_ft_cnn_model.ipynb')
    notebook = json.loads(notebook_path.read_text(encoding='utf-8'))

    assert notebook['nbformat'] == 4
    sources = [''.join(cell.get('source', [])) for cell in notebook['cells']]

    assert any('create_combined_model' in source for source in sources)
    assert any('make_gradcam_heatmap' in source for source in sources)
    assert any('stack_spatial_and_frequency_channels' in source for source in sources)
