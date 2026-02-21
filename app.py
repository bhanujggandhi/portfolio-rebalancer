"""
Portfolio Rebalancer — Entry Point
===================================
Run with:
    panel serve app.py --show
    python app.py
"""

import panel as pn
from   rebalancer.config        import APP_TITLE, PANEL_PORT
from   rebalancer.ui            import layout

pn.extension("tabulator", sizing_mode="stretch_width")

# Importing ui triggers widget creation and callback wiring.
# The layout object is what Panel serves to the browser.

layout.servable(title=APP_TITLE)

if __name__ == "__main__":
    pn.serve(layout, port=PANEL_PORT, show=True, title=APP_TITLE)
