
def raw_umap(
            ncols = 8,
            frameon=False, 
            color_map = 'Reds', 
            add_outline = True, 
            outline_color = ('white','lightgrey'), 
            outline_width = (0, 0.5), 
            size = 30):

    return dict(
        ncols = ncols, frameon = frameon, color_map = color_map,
        add_outline = add_outline, outline_color = outline_color, 
        outline_width = outline_width, size = size
    )

def topic_umap(
    ncols = 8,
    frameon = False,
    color_map = 'inferno',
):
    return dict(ncols = ncols, frameon = frameon, color_map = color_map)