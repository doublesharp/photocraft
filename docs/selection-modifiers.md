# Selection modifier keys

PhotoCraft uses Shift to add to the current selection, Option (Alt) to subtract,
and Shift + Option to intersect. This applies to marquee, lasso, polygonal and
magnetic lasso, Magic Wand, Quick Selection and Object Selection, including the
fill-area lasso in Content-Aware Fill.

The operation is captured when a gesture starts. Releasing its modifier before
releasing the pointer does not change the operation; the cursor badge continues
to show the captured mode. Quick Selection keeps its unmodified Add behavior.
Command retains the existing temporary Move behavior and command shortcuts.
Accessibility sticky Shift and Option provide the same selection overrides.

Automation can use the existing selection command `mode` parameter, including
`select.quick {"mode":"intersect", ...}`. In the Content-Aware Fill workspace,
`edit.contentAwareFill {"ui":{"lasso":{"points":[[x,y],...],"shift":true,
"alt":true}}}` applies the same intersection operation to the fill area.

The modifier correction is independent of local AI Smart Fill and model setup.
