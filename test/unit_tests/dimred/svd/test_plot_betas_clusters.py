import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from lasso.dimred.svd.plot_beta_clusters import plot_clusters_js


class TestBetaViz(TestCase):
    def setUp(self):
        self.betas = [np.array([[1, 1, 1], [1, 2, 3]])]
        self.ids = np.array([["sample0", "sample1"]])

    def test_plot_clusters_js(self):
        """Verify the generated HTML and plot data without pinning the bundled JavaScript."""
        html_page_str = plot_clusters_js(
            self.betas,
            self.ids,
            "",
            mark_timestamp=False,
            filename="testpage",
            write=False,
            show_res=False,
        )

        self.assertIsInstance(html_page_str, str)
        for snippet in (
            "<html>",
            "<title>3D Beta Embedding</title>",
            '<script type="text/javascript">',
            "const runIDs = [['0', '1']]",
            "name: 'cluster 0'",
            "text: ['sample0', 'sample1']",
            "x: [100.0, 100.0]",
            "y: [100.0, 200.0]",
            "z: [100.0, 300.0]",
            "traceList = [trace0, ]",
            "new THREE.WebGLRenderer",
            "</html>",
        ):
            with self.subTest(snippet=snippet):
                self.assertIn(snippet, html_page_str)

    def test_plot_clusters_js_with_outliers(self):
        betas = [np.array([[-3, 0, 1]]), np.array([[1, 2, 3]])]
        ids = np.array([["sample0"], ["sample1"]])

        html_page_str = plot_clusters_js(
            betas, ids, "", img_path="images", mark_outliers=True, write=False, show_res=False
        )

        for snippet in (
            'path = "images"',
            "const runIDs = [['0'], ['1']]",
            "name: 'outliers'",
            "color: 'black'",
            "text: ['sample0']",
            "x: [-300.0]",
            "y: [0.0]",
            "z: [100.0]",
            "name: 'cluster 1'",
            "text: ['sample1']",
            "x: [100.0]",
            "y: [200.0]",
            "z: [300.0]",
            "traceList = [trace0, trace1, ]",
        ):
            with self.subTest(snippet=snippet):
                self.assertIn(snippet, html_page_str)

    def test_plot_clusters_js_write(self):
        expected_html = plot_clusters_js(self.betas, self.ids, "", write=False, show_res=False)

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch("lasso.dimred.svd.plot_beta_clusters.webbrowser.open") as open_browser,
        ):
            result = plot_clusters_js(
                self.betas,
                self.ids,
                tmpdir,
                mark_timestamp=False,
                filename="testpage",
                show_res=False,
            )

            self.assertIsNone(result)
            self.assertEqual(
                Path(tmpdir, "testpage.html").read_text(encoding="utf-8"), expected_html
            )
            open_browser.assert_not_called()
