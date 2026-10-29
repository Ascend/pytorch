# Native API Format and Writing Guide

This guide was established in October 2026.

## 1. Distinguish Page Types First

| Page | Format |
| --- | --- |
| Module reference pages such as `pytorch_<version>/torch*.md` | Page description, optional TOC, category headings, and API entries |
| `overview.md` | Scope and description of the support list of this version; no API entry structure is imposed |
| `PyTorch-*.md` and `_menu_pt_native_apis.md` | Version entry or navigation; follow the actual menu conventions |
| Graph mode, DTensor, deterministic computation, and private API lists and appendixes | Keep their own lists, tables, or operation instructions; do not convert them into product support entries in bulk |

## 2. Module Page Structure

The names, hierarchy, and arrangement order of API modules and subheadings must align with the upstream PyTorch documentation of the corresponding version, and the in-page TOC must remain consistent with the body text. Keep the upstream heading names; do not translate, rename, merge categories, or reorder them alphabetically, by product, or by support status. When verifying, use the module or topic page of the target version, and do not substitute pages of other versions or the latest page. Product support fields, constraint notes, and site HTML wrappers follow the format of this repository; upstream is not required to have these fields.

The typical order is: top-level module heading, NOTE note, HTML TOC container, hidden module heading, categories, and entries.

- Use the actual module name for the top-level heading, for example, `# torch.nn.functional`.
- The NOTE explains the meaning of Constraints & Notes, that APIs not included in the list have not been verified, the product scope, and the upstream module link. When editing an existing page, keep the existing notes; for a new page, take the structure from a similar module of the same version and adjust the links; do not copy the wrong version.
- Keep the existing `div` style and the internal TOC heading for the TOC container. TOC entries point to category headings; keep the original hierarchy and anchor rules.
- Hidden headings like the following are part of the page structure; do not delete them as redundant headings or remove the zero-width character entity:

```html
<div style="display:none;">

## &#8203;torch.nn.functional

</div>
```

- Categories usually use second-level headings, and API entries usually use third-level headings. Existing classes and methods may carry `_`, `class`, or HTML decorations; keep them as in the target page and do not change them all into plain function headings.
- Use field labels inside API entries; do not add headings at the same level as the next API.

## 3. API Entry Structure

The following is a structural example. Content in angle brackets must be replaced; it does not express the support conclusion of any actual API.

```markdown
### <API fully qualified name>

<div style="margin-left: 2em">

**Native Documentation**: [<API name>](<verified link of the corresponding version>)

**Support Status**:

<!-- npu="910b" id<unused number in this file> -->
- <term>Atlas A2 training products</term>: <verified support status>
<!-- end id<same number> -->
<!-- npu="A3" id<another unused number> -->
- <term>Atlas A3 training products</term>: <verified support status>
<!-- end id<same number> -->
<!-- npu="950" id<another unused number> -->
- <term>Ascend 950DT products</term>: <verified support status>
<!-- end id<same number> -->

**Constraints & Notes**:

- <constraint or difference with evidence>

</div>
```

- The field order is Native Documentation, Support Status, and the optional Constraints & Notes. If there is no constraint information, omit the entire Constraints & Notes field; do not fill in "None" or leave an empty placeholder.
- A short constraint can be placed on the same line as the field: `` **Constraints & Notes**: `input` only supports fp16 and fp32 ``. For multiple constraints, put the field first, followed by a blank line and a list.
- Do not rewrite native APIs into a complete manual with "function description, parameters, and return values"; usage is provided through the Native Documentation link.
- Keep reasonable blank lines before and after code fences, lists, and HTML containers to prevent Markdown from being treated as plain text inside HTML.

## 4. Product Support Lists and Conditional Comments

Product filtering tags are used to filter displayed information by product. They are used line by line and do not support inline usage. The start comment and end comment each occupy a single line, and the text to be filtered is placed between them. Do not insert tags into part of a sentence, a list item, or a table cell to filter that part. This refers to `<!-- npu="..." id... -->` / `<!-- end id... -->`; `<term>` product name tags are not required to occupy a single line.

Correct example:

```markdown
<!-- npu="910b" id1 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id1 -->
```

Incorrect example (inline filtering is not supported):

```markdown
- Support Status: <!-- npu="910b" id1 --><term>Atlas A2 training products</term>: Supported<!-- end id1 -->
```

If a sentence contains different constraints for multiple products and each needs to be filtered separately, split them into complete lines or list items and wrap each with tags separately; keep the common conditions and the original semantics.

The current sample mapping is `910b` → Atlas A2 training products, `A3` → Atlas A3 training products, and `950` → Ascend 950DT products. Case is meaningful. For future products, confirm the codes from the samples of the target branch or processing tools; do not invent codes by yourself.

- The start comment and end comment of each pair must use the same ID. When editing an existing entry, do not change existing IDs or move comment boundaries to expand or narrow the scope.
- For new blocks, first scan the existing IDs in the whole file and assign an unused number; do not copy the ID of an adjacent entry. Existing samples are numbered by file; check for conflicts within the same file. If repository tools have additional scope requirements, follow the tools.
- Write the three product statuses in their respective blocks; do not merge them into one line to save space. Both supported and unsupported verified product records should be kept.
- `<term>` is a structural tag for product names; keep it. Existing names may have differences in spaces or suffixes; text polishing should not bulk-replace product terms.
- Do not automatically add tags to existing lists without comments; for new entries, prefer the mainstream structure of the same page. Migrate historical support tables only when the user explicitly requests it.
- If the target page also uses conditional comments in the constraint text, keep the original scope and list hierarchy. For product-specific constraints without conditional comments, clearly state the applicable product at the beginning of the sentence; do not assume the text will be filtered just because the support list has filter blocks.
- If no release processing tool is found, you can only confirm the comment structure; do not claim that the rendering result after product filtering has been verified.

## 5. Expressing Constraints and Notes

### Parameters, dtypes, and Values

- Use backticks for parameter names, attribute names, API names, and code expressions; express `True`, `False`, `None`, and pattern strings as code values.
- Constraints should clearly identify the object. Functions do not default to `input`; Tensor methods may involve `self`; creation APIs may involve `dtype`; for multi-input APIs, write them separately according to the actual objects. If the object is unclear, verify the signature and support evidence; do not guess based only on the first parameter.
- Keep the existing dtype set and naming of this version; do not uniformly extend them to `torch.*`, and do not copy or deduplicate cross-parameter constraints from other versions.
- Use "only supports" for a clear and complete scope; do not mechanically add "only" when the original text merely says "supports".
- Keep the original values and units. When converting double negations into positive constraints, verify the inequality direction, divisibility, whether boundaries are included, and the logical relationship among multiple conditions.
- Do not change "may fall back" to "will fall back", do not turn precision differences into errors, and do not add unverified implementation reasons.

### Simple and Complex Entries

A simple entry writes one constraint per item, for example, "`input` only supports 1 to 8 dimensions". Complex entries can be grouped by input specifications, parameter combinations, execution modes, native differences, and workaround methods; use only the groups that are actually needed.

For example, an SDPA entry involves multiple input dtypes, layouts, head count relationships, dimension ranges, mask and causal combinations, backend settings, and native differences. Keep its hierarchy: use parent items for common conditions and child items for specific constraints. Do not flatten it into one long paragraph, and do not require every API to have these groups.

- Constraints under the same parent item are easily understood to hold simultaneously by default; explicitly state "or" cases.
- Qualifiers related to products, dtypes, forward/backward direction, or execution modes should be placed close to the corresponding constraints.
- Repeated descriptions can be merged only when the objects, conditions, and technical conclusions are exactly the same; keep separate constraints on different parameters even if they look similar.
- For complex constraints, do not add fourth-level headings that disrupt the entry structure; prefer lists and bold group labels.

## 6. Examples

Add examples when they can demonstrate switch configuration, context, parameter combinations, or verified workaround operations. Pure dtype lists, scope statements, and precision risks can usually be expressed in text; do not force code to satisfy a template.

- Indent the code block within the list item that introduces the example; keep the opening and closing fences consistent, use actual language markers such as `python`, and leave blank lines around the fences.
- New examples include necessary imports, prerequisite configuration, and input definitions. If only operation snippets are provided, clearly state which inputs are prepared by the caller; such snippets cannot be called independently runnable examples.
- Code must comply with the version, supported products, dtypes, shapes, and execution modes of this entry. Upstream examples cannot directly prove that NPU examples are valid.
- Examples that set a switch should set it before the corresponding operation; when global behavior is involved, state its known scope of effect and do not guess when it takes effect.
- You may improve the imports and formatting of existing examples, but do not rewrite the technical path for formatting reasons. When the NPU runtime environment is unavailable, verify statically; do not report that the run passed.

## 7. Versions, Links, and Navigation

- Keep the display scope consistent with the upstream PyTorch of the corresponding version: modules, categories, and APIs that upstream has deprecated or no longer displays should not be displayed in this list either. When maintaining the display scope, remove the corresponding body entries and clean up the related in-page TOC and navigation entries in sync; do not keep entries displayed just because the old links are still accessible. Base the decision on the deprecation notes and display status of the target version upstream; do not judge solely by link access failures, and do not apply changes of new versions to historical versions.
- A directory such as `pytorch_2-14-0` corresponds to page version 2.14.0; common PyTorch documentation URLs currently use `/docs/2.14/`. Use the actual links of the target APIs; do not mechanically concatenate them.
- Class methods, module entry points, and external modules such as torchvision may use special URLs or anchors; confirm the links that need to be modified item by item. Deprecated interfaces are handled according to the display scope rules; do not keep them displayed by fixing links.
- Entries in topics such as Named Tensors should link to the corresponding topic page and anchor; do not change them into ordinary API links based on the displayed API name. For example, in the existing 2.11 documentation, `torch.Tensor` under Named Tensors points to `named_tensor.html#named-tensors`, and `torch.Tensor.names` points to `named_tensor.html#torch.Tensor.names`; do not replace them with `generated/torch.Tensor.html`, `generated/torch.Tensor.names.html`, or an ordinary Tensor reference page. For other versions, verify the actual topic paths and anchors of that version; do not mechanically copy these examples.
- Links of APIs with the same name in different topics may differ; when checking links, consider both the topic and the target description, not just the API name. Batch link fixes follow this rule as well.
- Place new APIs in the categories and relative positions of the corresponding version upstream, keeping the names, structure, and order of modules and subheadings consistent with upstream; when adding or adjusting categories or pages, update the involved TOC or navigation in sync; do not force an in-page TOC entry for every API.
- For a new version, check the version entry, overview, and the overall menu; do not claim support facts are consistent after only batch-replacing version numbers.
- Do not confuse not included, not verified, and not supported. Graph mode, DTensor, and deterministic computation support are independent capabilities; they cannot be inferred from ordinary product support status.

## 8. Minimal Acceptance

Text polishing: review the diff and confirm that facts, product blocks, IDs, links, and heading anchors are preserved. New entries: verify the evidence, field order, container closure, non-conflicting IDs, category positions, and related navigation. Structural migration: additionally check the product status and constraint attribution of each API before and after the migration.

Verification should be performed according to the specific scope of modification; do not modify unrelated pages just because the repository has historical warnings. Distinguish static structure checks, documentation tool checks, site rendering, and NPU example execution, and report honestly which ones were completed.
