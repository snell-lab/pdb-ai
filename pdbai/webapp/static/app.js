// The PDB-AI page script. It reads the configuration that the server embeds in #pdbaiConfig, once. It keeps the
// build form consistent with the chosen technique and profile, and it starts builds and column filters. It
// follows a build's progress and keeps the Runs panel current. Every string from the server reaches the page as
// text, never as HTML. The page hides an element only through the hidden attribute. The file reads top down,
// from the configuration and helpers to the renderers, the event handlers, and init() at the bottom.
(() => {
  'use strict';

  const config = JSON.parse(document.getElementById('pdbaiConfig').textContent);
  const urls = config.urls;
  const techniqueConfig = config.techniques;

  // This pattern must stay identical to PDB_ID_DELIMITERS in pdbai/webapp/options.py, or the page would count
  // identifiers the server then refuses. The whitespace is spelled out because JavaScript's \s also matches a
  // non-breaking space, which the server keeps inside an identifier.
  const PDB_ID_DELIMITERS = /[\t\n\v\f\r ,;]+/;
  // This is the identifier rule the server enforces, in pdbai/identifiers.py.
  const PDB_ID_PATTERN = /^[1-9][0-9A-Za-z]{3}$/;
  // These are the delimiters the server splits column lists on, in pdbai/column_filtering.py.
  const COLUMN_LIST_DELIMITERS = /[\r\n,;]+/;
  const STATUS_POLL_MILLISECONDS = 900;
  // A dev-server reload or a dropped connection answers a poll or two with an error. A failed status poll is
  // therefore retried this many times before the page gives up on the run. Each retry waits twice as long as the
  // one before.
  const STATUS_POLL_RETRIES = 4;
  // The Runs panel polls quickly while a run is building, and slowly otherwise. The slow poll still notices runs
  // that another tab or the command line starts.
  const RUNS_POLL_LIVE_MILLISECONDS = 1500;
  const RUNS_POLL_IDLE_MILLISECONDS = 8000;
  const FILE_LIST_LIMIT = 6;
  const RUN_ACTIONS = {
    pause: { template: urls.runPause, method: 'POST', verb: 'paused' },
    resume: { template: urls.runResume, method: 'POST', verb: 'resumed' },
    delete: { template: urls.runDelete, method: 'DELETE', verb: 'deleted' },
  };
  const SHORT_RUN_ID_LENGTH = 12;

  const buildForm = document.getElementById('dataset-form');
  const buildSubmitButton = buildForm.querySelector('[type="submit"]');
  const pdbIdsInput = document.getElementById('pdbIds');
  const pdbIdsHint = document.getElementById('pdbIdsHint');
  const selectButton = document.getElementById('selectFromPdbButton');
  const selectHint = document.getElementById('selectHint');
  const mmcifFilesInput = document.getElementById('mmcifFiles');
  const dropzone = document.getElementById('mmcifDropzone');
  const fileList = document.getElementById('mmcifFileList');
  const techniqueSelect = document.getElementById('technique');
  const xraySourceField = document.getElementById('xraySourceField');
  const xraySourceSelect = document.getElementById('xraySource');
  const columnFilterDetails = document.getElementById('columnFilterDetails');
  const profileSelect = document.getElementById('columnProfile');
  const profileDescription = document.getElementById('profileDescription');
  const columnFilterModeSelect = document.getElementById('columnMode');
  const columnFilterListInput = document.getElementById('columnFilterList');
  const enginesDetails = document.getElementById('enginesDetails');
  const engineSelectAllButton = document.getElementById('engineSelectAllButton');
  const datasetShapeSelect = document.getElementById('datasetShape');
  const statusPanel = document.getElementById('status');
  const statusText = document.getElementById('status-text');
  const statusProgress = document.getElementById('status-progress');
  const progressFill = document.getElementById('progress-bar');
  const progressMeta = document.getElementById('progress-meta');
  const buildResultsSection = document.getElementById('results');

  const filterForm = document.getElementById('filter-form');
  const filterSubmitButton = filterForm.querySelector('[type="submit"]');
  const filterFileInput = document.getElementById('filterFile');
  const filterModeSelect = document.getElementById('filterMode');
  const filterColumnsInput = document.getElementById('filterColumns');
  const filterStatus = document.getElementById('filter-status');
  const filterResultsSection = document.getElementById('filter-results');

  const runsRefreshButton = document.getElementById('runsRefreshButton');
  const runsAutoRefreshCheckbox = document.getElementById('runsAutoRefresh');
  const runsEmpty = document.getElementById('runs-empty');
  const runsTable = document.getElementById('runs-table');
  const runsTableBody = runsTable.querySelector('tbody');
  const runsStatus = document.getElementById('runs-status');

  // Every engine setting is rendered from its declaration with data-option. An engine's switch carries
  // data-engine-toggle. A setting that refines a flag names that flag in data-refines.
  const optionInputs = Array.from(document.querySelectorAll('[data-option]'));
  const flagInputs = optionInputs.filter((input) => input.dataset.kind === 'flag');
  const engineCheckboxes = optionInputs.filter((input) => input.hasAttribute('data-engine-toggle'));
  const countInputs = optionInputs.filter((input) => input.dataset.kind === 'count');
  // Every range filter of the archive search is rendered from its declaration. Its field carries
  // data-search-filter and is named by its request key. The menu holds a checkbox for it, which names the same key
  // in data-search-filter-toggle, and the field shows only while that checkbox is ticked.
  const searchFilterMenu = document.getElementById('searchFilterMenu');
  const searchFilterMenuLabel = document.getElementById('searchFilterMenuLabel');
  const searchFilterMenuDefaultLabel = searchFilterMenuLabel.textContent;
  const searchFiltersSection = document.getElementById('searchFilters');
  const searchFilterInputs = Array.from(document.querySelectorAll('[data-search-filter]'));
  const searchFilterToggles = Array.from(document.querySelectorAll('[data-search-filter-toggle]'));
  const searchQuantityNotes = Array.from(document.querySelectorAll('[data-search-quantity-note]'));

  const techniqueLabels = Object.fromEntries(
    techniqueConfig.techniques.map((technique) => [technique.id, technique.displayLabel]),
  );
  const xraySourceLabels = Object.fromEntries(
    techniqueConfig.xraySources.sources.map((source) => [source.id, source.displayLabel]),
  );
  // The X-ray source value that admits every source, sent whenever the filter does not apply.
  const allXraySources = techniqueConfig.xraySources.all;
  // Each flag maps to the set of techniques it supports, or to null when it applies to every technique.
  const techniquesByToggle = Object.fromEntries(
    techniqueConfig.compatibility.map((record) => [
      record.toggle,
      record.techniques === null ? null : new Set(record.techniques),
    ]),
  );
  const profilesById = new Map(config.profiles.map((profile) => [profile.id, profile]));
  const hintDefaults = new Map([
    [pdbIdsHint, pdbIdsHint.textContent],
    [selectHint, selectHint.textContent],
  ]);

  // This is the archive-search record behind the identifier box. A build sends it only while the box holds exactly
  // the identifiers the search filled in. Changing the technique or X-ray source drops it, because the record must
  // name the technique and source the run builds with. Changing a range filter does not, because the record still
  // describes the search that filled the box.
  let lastSelection = null;
  let lastSelectionText = '';
  // This is what applyProfile set. A build sends profileId only while the form still holds it, because otherwise
  // the server would apply the profile again over the user's later edits.
  let appliedProfile = null;
  // This is the run the status panel follows. Each followRun call takes a new generation. A response to an older
  // poll is then dropped, not drawn over the newer run.
  let followedRunId = null;
  let statusPollGeneration = 0;
  let statusPollTimer = null;
  let runsSnapshot = [];
  let runsRequestInFlight = false;
  let runsRefreshQueued = false;
  let runsPollTimer = null;
  // These are the run IDs with a pause, resume, or delete request in flight.
  const pendingRunActions = new Set();
  let pointerDownOnRuns = false;
  // This says whether the runs status line shows a listing error. The next successful listing clears that error.
  let runsStatusShowsListingError = false;

  // The helpers below are shared by the renderers and the event handlers.

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    // Status and report values can come from outside this process, so they are only ever set as text.
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function plural(count, singular, pluralForm) {
    return `${Number(count).toLocaleString()} ${count === 1 ? singular : pluralForm}`;
  }

  function splitIds(raw) {
    return raw.split(PDB_ID_DELIMITERS).filter(Boolean);
  }

  function normalizeIdText(raw) {
    return splitIds(raw)
      .map((id) => id.toUpperCase())
      .join(',');
  }

  function expandTemplate(template, runId) {
    // A run ID fills one path segment and must not change the route it is put into.
    return template.replace('RUN_ID', encodeURIComponent(runId));
  }

  function formatTimestamp(epochSeconds) {
    if (typeof epochSeconds !== 'number') return 'n/a';
    const date = new Date(epochSeconds * 1000);
    return Number.isNaN(date.getTime()) ? 'n/a' : date.toLocaleString();
  }

  // Return the JSON a request answers. Any failure throws an Error whose message starts with failureMessage. The
  // error's status is the HTTP status, or 0 when no answer arrived.
  async function fetchJson(url, options, failureMessage) {
    let response;
    try {
      response = await fetch(url, options);
    } catch {
      throw Object.assign(new Error(`${failureMessage} The server could not be reached.`), { status: 0 });
    }
    const data = await response.json().catch(() => null);
    if (!response.ok) {
      // A proxy or a crashed server answers with HTML. The HTTP status then stands in for the missing JSON error.
      const detail =
        data && typeof data.error === 'string' ? data.error : `The server answered HTTP ${response.status}.`;
      throw Object.assign(new Error(`${failureMessage} ${detail}`), { status: response.status });
    }
    if (data === null) {
      throw Object.assign(new Error(`${failureMessage} The server's answer was not JSON.`), {
        status: response.status,
      });
    }
    return data;
  }

  function setHint(hint, message = hintDefaults.get(hint), isError = false) {
    hint.textContent = message;
    hint.classList.toggle('is-error', isError);
  }

  function setInvalid(control, invalid) {
    control.classList.toggle('is-error', invalid);
    if (invalid) {
      control.setAttribute('aria-invalid', 'true');
    } else {
      control.removeAttribute('aria-invalid');
    }
  }

  function setStatus(panel, message, isError = false, textElement = panel) {
    panel.hidden = false;
    panel.classList.toggle('is-error', isError);
    textElement.textContent = message;
  }

  function setChecked(checkbox, checked) {
    if (checkbox.checked === checked) return;
    checkbox.checked = checked;
    // Dependent captions and controls listen for the same event a click sends.
    checkbox.dispatchEvent(new Event('change', { bubbles: true }));
  }

  function allChecked(checkboxes) {
    return checkboxes.length > 0 && checkboxes.every((checkbox) => checkbox.checked);
  }

  function isAcceptedUpload(name) {
    const lowered = name.toLowerCase();
    return config.acceptedSuffixes.some((suffix) => lowered.endsWith(suffix));
  }

  function hasColumns(text) {
    return text.split(COLUMN_LIST_DELIMITERS).some((name) => name.trim());
  }

  // The server requires a filter type and a column list to come together. Return the control that breaks this
  // rule, with a message saying how to fix it, or null when the pair is consistent.
  function columnFilterProblem(modeSelect, listInput) {
    const listed = hasColumns(listInput.value);
    if (modeSelect.value && !listed) {
      return {
        control: listInput,
        message: 'List the columns to include or exclude, or set the filter type to no column filtering.',
      };
    }
    if (!modeSelect.value && listed) {
      return {
        control: modeSelect,
        message: 'Choose whether to include or exclude the listed columns, or clear the list.',
      };
    }
    return null;
  }

  function xraySourceApplies() {
    return techniqueSelect.value === techniqueConfig.xraySources.technique;
  }

  function currentXraySource() {
    return xraySourceApplies() ? xraySourceSelect.value : allXraySources;
  }

  function toggleSupportsTechnique(toggle, technique) {
    const supported = techniquesByToggle[toggle];
    return supported === null || supported.has(technique);
  }

  function optionInput(name) {
    return optionInputs.find((input) => input.dataset.option === name);
  }

  function availableEngineCheckboxes() {
    // A disabled checkbox belongs to an engine the selected technique rules out.
    return engineCheckboxes.filter((checkbox) => !checkbox.disabled);
  }

  // Return the engine settings under the keys the server's API expects. A refining setting whose switch is off is
  // left out, so the server uses its default. A count is sent as the text typed, and the server checks it.
  function collectOptionValues() {
    const values = {};
    optionInputs.forEach((input) => {
      if (input.dataset.refines && input.disabled) return;
      const key = input.name;
      if (input.dataset.kind === 'flag') {
        values[key] = input.checked;
      } else if (input.dataset.kind === 'count') {
        if (input.value) values[key] = input.value;
      } else {
        values[key] = input.value;
      }
    });
    return values;
  }

  function searchFilterField(input) {
    return input.closest('.search-filters__field');
  }

  function chosenSearchFilterInputs() {
    return searchFilterInputs.filter((input) => !searchFilterField(input).hidden);
  }

  // Return the range filters that are ticked and filled in, under the keys the server's API expects. Each is sent
  // as the text typed, and the server checks it.
  function collectSearchFilters() {
    const values = {};
    chosenSearchFilterInputs().forEach((input) => {
      if (input.value) values[input.name] = input.value;
    });
    return values;
  }

  // Bring the fields in line with the menu's checkboxes. The field of a ticked filter shows, with the note on its
  // quantity. The field of any other filter is hidden and emptied, so a value typed earlier is not sent.
  function syncSearchFilters() {
    const chosenQuantities = new Set();
    searchFilterInputs.forEach((input) => {
      const toggle = searchFilterToggles.find((box) => box.dataset.searchFilterToggle === input.name);
      const field = searchFilterField(input);
      field.hidden = !toggle.checked;
      if (toggle.checked) {
        chosenQuantities.add(field.dataset.searchQuantity);
      } else {
        input.value = '';
        setInvalid(input, false);
      }
    });
    searchQuantityNotes.forEach((note) => {
      note.hidden = !chosenQuantities.has(note.dataset.searchQuantityNote);
    });
    const chosen = chosenSearchFilterInputs().length;
    searchFiltersSection.hidden = chosen === 0;
    searchFilterMenuLabel.textContent = chosen
      ? `${searchFilterMenuDefaultLabel} (${chosen})`
      : searchFilterMenuDefaultLabel;
  }

  // Return the message that stops a search before it is sent, and mark the field at fault. Return '' when the
  // filters may be sent.
  function searchFilterProblem() {
    const chosen = chosenSearchFilterInputs();
    // A number input whose text is not a number reports an empty value, as a field left blank does. Either way
    // the search would silently run without a filter the form shows as ticked.
    const bad = chosen.find((input) => input.validity.badInput) || chosen.find((input) => !input.value.trim());
    if (!bad) return '';
    setInvalid(bad, true);
    bad.focus();
    const label = document.querySelector(`label[for="${bad.id}"]`).textContent;
    const wanted = bad.dataset.searchFilterKind === 'accessions' ? 'at least one accession' : 'a positive number';
    return `Enter ${wanted} for "${label}", or remove that filter.`;
  }

  function profileStillApplied() {
    return (
      appliedProfile !== null &&
      profileSelect.value === appliedProfile.id &&
      columnFilterModeSelect.value === appliedProfile.mode &&
      columnFilterListInput.value === appliedProfile.list &&
      appliedProfile.turnedOnCheckboxes.every((checkbox) => checkbox.checked)
    );
  }

  function buildSettings() {
    const settings = {
      ...collectOptionValues(),
      datasetShape: datasetShapeSelect.value,
      technique: techniqueSelect.value,
      xraySource: currentXraySource(),
    };
    if (profileStillApplied()) {
      // The server applies the profile's own filter, minus the columns of engines the technique rules out. The
      // form's copy of the filter still holds those columns. It is not sent, because a filter the request names
      // would win over the profile's.
      settings.profileId = appliedProfile.id;
    } else {
      settings.columnFilterMode = columnFilterModeSelect.value;
      settings.columnFilterList = columnFilterListInput.value;
    }
    return settings;
  }

  function anyRunLive() {
    // A run just submitted counts as live before its first listing shows it.
    return followedRunId !== null || runsSnapshot.some((run) => run.live);
  }

  // The renderers below draw part of the page from the form's state or from a server answer.

  function validatePdbIds() {
    const ids = splitIds(pdbIdsInput.value);
    const invalid = ids.filter((id) => !PDB_ID_PATTERN.test(id));
    setInvalid(pdbIdsInput, invalid.length > 0);
    if (!ids.length) {
      setHint(pdbIdsHint);
    } else if (invalid.length) {
      setHint(
        pdbIdsHint,
        `Not a PDB ID: ${invalid.join(', ')}. ` +
          'A PDB ID is a digit from 1 to 9 followed by three letters or digits.',
        true,
      );
    } else {
      setHint(pdbIdsHint, `${plural(ids.length, 'PDB ID', 'PDB IDs')} ready.`);
    }
    return invalid.length === 0;
  }

  function renderFileList() {
    const files = Array.from(mmcifFilesInput.files);
    fileList.replaceChildren(
      ...files.slice(0, FILE_LIST_LIMIT).map((file) => el('span', 'dropzone__file', file.name)),
    );
    if (files.length > FILE_LIST_LIMIT) {
      fileList.appendChild(el('span', 'dropzone__file', `+${files.length - FILE_LIST_LIMIT} more`));
    }
    fileList.hidden = files.length === 0;
  }

  function updateProfileDescription() {
    const profile = profilesById.get(profileSelect.value);
    const description = profile ? profile.description : '';
    profileDescription.textContent = description;
    profileDescription.hidden = !description;
  }

  // A setting that refines a flag only matters while that flag is on and usable. The page renders each refinement
  // after the flag it refines. One pass in page order therefore settles refinements of refinements too.
  function updateRefinements() {
    optionInputs.forEach((input) => {
      const parentName = input.dataset.refines;
      if (!parentName) return;
      const parent = optionInput(parentName);
      input.disabled = !(parent?.checked && !parent.disabled);
    });
  }

  function updateEngineSelectAllLabel() {
    const available = availableEngineCheckboxes();
    engineSelectAllButton.textContent = allChecked(available) ? 'Deselect All' : 'Select All';
    engineSelectAllButton.disabled = available.length === 0;
  }

  function setCheckboxAvailability(checkbox, available, techniqueLabel) {
    const label = checkbox.closest('label');
    let note = label.querySelector(':scope > .tag');
    label.classList.toggle('is-unavailable', !available);
    if (available) {
      checkbox.disabled = false;
      if (note) note.remove();
      return;
    }
    setChecked(checkbox, false);
    checkbox.disabled = true;
    // The note sits inside the label, so it is read as part of the checkbox's name.
    if (!note) {
      note = el('span', 'tag');
      label.appendChild(note);
    }
    note.textContent = `Not available for ${techniqueLabel}`;
  }

  function applyTechniqueCompatibility() {
    const technique = techniqueSelect.value;
    const techniqueLabel = techniqueLabels[technique] || technique;
    engineCheckboxes.forEach((checkbox) => {
      setCheckboxAvailability(
        checkbox,
        toggleSupportsTechnique(checkbox.dataset.option, technique),
        techniqueLabel,
      );
    });
    updateRefinements();
    updateEngineSelectAllLabel();
  }

  function applyXraySourceAvailability() {
    // The source filter applies only within X-ray diffraction. For any other technique it is reset and hidden,
    // so it can never be submitted.
    const applies = xraySourceApplies();
    if (!applies) xraySourceSelect.value = allXraySources;
    xraySourceSelect.disabled = !applies;
    xraySourceField.hidden = !applies;
  }

  function applyProfile(profileId) {
    const profile = profilesById.get(profileId);
    if (!profile) return;
    // A profile switches on what its columns need and leaves every other switch alone. The server and the command
    // line apply the same rule. A disabled engine cannot serve the profile under this technique.
    const toggles = new Set(profile.toggles);
    const turnedOn = availableEngineCheckboxes().filter((checkbox) => toggles.has(checkbox.dataset.option));
    turnedOn.forEach((checkbox) => setChecked(checkbox, true));
    columnFilterModeSelect.value = profile.filterMode || '';
    columnFilterListInput.value = profile.filterMode ? profile.columns.join('\n') : '';
    setInvalid(columnFilterModeSelect, false);
    setInvalid(columnFilterListInput, false);
    if (toggles.size) enginesDetails.open = true;
    appliedProfile = {
      id: profile.id,
      mode: columnFilterModeSelect.value,
      list: columnFilterListInput.value,
      turnedOnCheckboxes: turnedOn,
    };
  }

  function showBuildStatus(message) {
    setStatus(statusPanel, message, false, statusText);
  }

  function showBuildError(message) {
    setStatus(statusPanel, message, true, statusText);
    statusProgress.hidden = true;
  }

  function setProgressFraction(fraction) {
    const percent = Math.round(Math.min(1, Math.max(0, fraction)) * 100);
    progressFill.style.width = `${percent}%`;
    progressFill.setAttribute('aria-valuenow', String(percent));
  }

  function showProgress() {
    statusProgress.hidden = false;
    setProgressFraction(0);
    progressMeta.textContent = '';
  }

  function stageLabel(stage) {
    return config.stageLabels[stage] || stage;
  }

  function updateProgress(progress) {
    // The status line names no stage. An intake event reports a refused entry, and the build goes on after it.
    statusText.textContent = 'Building the dataset…';
    if (!progress) return;
    const current = Number(progress.current) || 0;
    const total = Number(progress.total) || 0;
    setProgressFraction(total > 0 ? current / total : 0);
    const parts = [stageLabel(progress.stage), progress.subject];
    if (total > 0) parts.unshift(`${current}/${total}`);
    progressMeta.textContent = parts.filter(Boolean).join(' · ');
  }

  function focusHeading(heading) {
    // Focus moves to new results unless the user is typing somewhere. Taking focus then would lose their place.
    if (document.activeElement?.matches('input, textarea, select')) return;
    heading.tabIndex = -1;
    heading.focus();
  }

  function downloadLink(href, text, className) {
    const link = el('a', className, text);
    link.href = href;
    link.download = '';
    return link;
  }

  function tableLabel(table) {
    return config.tableLabels[table] || `the ${table} table`;
  }

  function columnFilterSummaryLine(label, names, className) {
    const node = el('div', className);
    node.appendChild(el('strong', null, `${label}:`));
    node.appendChild(document.createTextNode(` ${names.length ? names.join(', ') : 'None'}`));
    return node;
  }

  function renderColumnFilterSummary(container, columnFilter) {
    if (!columnFilter?.mode) return;
    const matched = Array.isArray(columnFilter.matchedColumns) ? columnFilter.matchedColumns : [];
    const modeLabel = columnFilter.mode === 'include' ? 'Included columns' : 'Excluded columns';
    container.appendChild(columnFilterSummaryLine(modeLabel, matched, 'column-filter-summary'));
    if (Array.isArray(columnFilter.missingColumns) && columnFilter.missingColumns.length) {
      container.appendChild(
        columnFilterSummaryLine(
          'Missing columns',
          columnFilter.missingColumns,
          'column-filter-summary column-filter-summary--missing',
        ),
      );
    }
  }

  // Draw one card per count the server sent. A count missing from the summary gets no card, not a card that
  // reads "NaN".
  function renderCards(container, cards) {
    container.replaceChildren(
      ...cards
        .filter((card) => typeof card.value === 'number')
        .map((card) => {
          const article = el('article', 'summary-card');
          article.appendChild(el('h3', null, card.label));
          article.appendChild(el('p', 'summary-card__value', card.value.toLocaleString()));
          article.appendChild(el('p', 'summary-card__meta', card.description));
          return article;
        }),
    );
  }

  function renderWarnings(container, warnings) {
    const list = Array.isArray(warnings) ? warnings : [];
    container.hidden = list.length === 0;
    const items = el('ul');
    list.forEach((warning) => items.appendChild(el('li', null, warning)));
    container.replaceChildren(el('h3', null, 'Warnings'), items);
  }

  function renderPreview(container, rows) {
    const list = Array.isArray(rows) ? rows : [];
    container.hidden = list.length === 0;
    container.replaceChildren();
    if (!list.length) return;
    const count = plural(list.length, 'row', 'rows');
    const title = list.length < config.previewRows ? `Preview (all ${count})` : `Preview (first ${count})`;
    const columns = Object.keys(list[0]);
    const headerRow = el('tr');
    columns.forEach((column) => headerRow.appendChild(el('th', null, column)));
    const head = el('thead');
    head.appendChild(headerRow);
    const body = el('tbody');
    list.forEach((row) => {
      const tableRow = el('tr');
      columns.forEach((column) => tableRow.appendChild(el('td', null, row[column] ?? '')));
      body.appendChild(tableRow);
    });
    const table = el('table');
    table.append(head, body);
    const scroller = el('div', 'preview__scroller');
    scroller.appendChild(table);
    container.append(el('h3', null, title), scroller);
  }

  // Fill a result panel from a server answer and show it. The build and the filter tool answer with the same
  // columnFilter, warnings, and preview fields, and their panels share one structure.
  function showResultPanel(section, links, cards, data) {
    const download = section.querySelector('.download');
    download.replaceChildren(...links);
    renderColumnFilterSummary(download, data.columnFilter);
    renderCards(section.querySelector('.summary-cards'), cards);
    renderWarnings(section.querySelector('.warnings'), data.warnings);
    renderPreview(section.querySelector('.preview'), data.preview);
    section.hidden = false;
    focusHeading(section.querySelector('h2, h3'));
  }

  function renderBuildResults(result, runId) {
    const additionalFiles = result.additionalCsvUrls || {};
    const links = [];
    if (Object.keys(additionalFiles).length) {
      const archiveUrl = expandTemplate(urls.runDownloadAll, runId);
      links.push(downloadLink(archiveUrl, 'Download All (.zip)', 'primary'));
      // An engine's own tables can follow a flat table too. The primary table has one row per atom in either
      // shape, so it takes the atoms label.
      links.push(downloadLink(result.csvUrl, `Download ${tableLabel('atoms')}`, 'ghost'));
      Object.entries(additionalFiles).forEach(([table, href]) => {
        links.push(downloadLink(href, `Download ${tableLabel(table)}`, 'ghost'));
      });
    } else {
      links.push(downloadLink(result.csvUrl, 'Download CSV', 'primary'));
    }
    if (result.runReportUrl) links.push(downloadLink(result.runReportUrl, 'Download run report', 'ghost'));
    if (result.runSummaryUrl) links.push(downloadLink(result.runSummaryUrl, 'Download run summary', 'ghost'));

    const summary = result.summary || {};
    const requested =
      typeof summary.requestedCount === 'number' ? ` of ${summary.requestedCount.toLocaleString()} requested` : '';
    const cards = [
      { label: 'PDB entries', value: summary.pdbCount, description: `Structures built${requested}` },
      { label: 'Atom-site rows', value: summary.atomCount, description: 'Rows in the generated dataset' },
      { label: 'Columns', value: summary.columnCount, description: 'Columns available in the dataset' },
    ];
    showResultPanel(buildResultsSection, links, cards, result);
  }

  function renderFilterResults(data) {
    const links = [downloadLink(data.csvUrl, 'Download filtered CSV', 'primary')];
    const summary = data.summary || {};
    const cards = [
      { label: 'Rows', value: summary.rowCount, description: 'Rows after filtering' },
      { label: 'Columns', value: summary.columnCount, description: 'Columns in the filtered dataset' },
    ];
    showResultPanel(filterResultsSection, links, cards, data);
  }

  function setRunsStatus(message, isError = false) {
    setStatus(runsStatus, message, isError);
    runsStatusShowsListingError = false;
  }

  function isHeldElsewhere(run) {
    // When another process, such as the command line, is building a run, the server reports it as running but
    // not live.
    return run.status === 'running' && !run.live;
  }

  function isUploadRun(run) {
    return run.settings?.source === 'upload';
  }

  function statusCell(run) {
    const cell = el('td');
    const status = run.status || 'unknown';
    cell.appendChild(
      el('span', `run-badge run-badge--${status.replace(/[^a-z_]/gi, '')}`, status.replace(/_/g, ' ')),
    );
    if (isHeldElsewhere(run)) cell.appendChild(el('div', 'run-meta', 'Being built by another process.'));
    if (run.error) cell.appendChild(el('div', 'run-meta', run.error));
    return cell;
  }

  function progressCell(run) {
    const counts = run.counts || {};
    const total = Number(counts.total) || 0;
    const done = Number(counts.done) || 0;
    const failed = Number(counts.failed) || 0;
    const refused = Number(counts.refused) || 0;
    const settled = done + failed + refused;
    const lastProgress = run.lastProgress;

    const fill = el('span', 'progress__fill');
    fill.style.width = `${total > 0 ? Math.min(100, (settled / total) * 100) : 0}%`;
    const bar = el('div', 'progress progress--compact');
    bar.appendChild(fill);

    let meta = `${done}/${total} done`;
    if (failed) meta += ` · ${failed} failed`;
    if (refused) meta += ` · ${refused} refused`;
    // The last event of a settled run describes finished work, so only a running run shows it.
    if (run.status === 'running' && lastProgress && lastProgress.stage) {
      meta += ` · ${[stageLabel(lastProgress.stage), lastProgress.subject].filter(Boolean).join(' ')}`;
    }
    const wrap = el('div', 'run-progress');
    wrap.append(bar, el('div', 'run-progress__meta', meta));
    const cell = el('td');
    cell.appendChild(wrap);
    return cell;
  }

  function sourceLabel(run) {
    const settings = run.settings || {};
    let label = 'PDB IDs';
    if (isUploadRun(run)) {
      label = 'mmCIF upload';
    } else if (run.selection) {
      label = 'PDB search';
    }
    if (!settings.technique) return label;
    label += ` · ${techniqueLabels[settings.technique] || settings.technique}`;
    const source = settings.xray_source;
    if (source && source !== allXraySources) label += ` · ${xraySourceLabels[source] || source}`;
    return label;
  }

  function actionsCell(run) {
    const cell = el('td', 'run-actions');
    const runId = run.runId;
    const pending = pendingRunActions.has(runId);
    const addButton = (action, className, label) => {
      const button = el('button', `${className} small`, label);
      button.type = 'button';
      button.dataset.action = action;
      button.dataset.runId = runId;
      button.disabled = pending;
      button.setAttribute('aria-label', `${label} run ${runId}`);
      cell.appendChild(button);
    };
    const resumable = ['paused', 'error', 'stale'].includes(run.status) && !isUploadRun(run);
    if (run.live) {
      addButton('pause', 'ghost', 'Pause');
    } else if (resumable) {
      addButton('resume', 'primary', 'Resume');
    }
    if (run.hasDataset) {
      const link = downloadLink(expandTemplate(urls.runDownloadAll, runId), 'Download', 'ghost small');
      link.setAttribute('aria-label', `Download run ${runId}`);
      cell.appendChild(link);
    }
    if (!run.live && !isHeldElsewhere(run)) addButton('delete', 'danger', 'Delete');
    return cell;
  }

  function runRow(run) {
    const runId = String(run.runId);
    const shortRunId = runId.length > SHORT_RUN_ID_LENGTH ? `${runId.slice(0, SHORT_RUN_ID_LENGTH)}…` : runId;
    const idNode = el('div', 'run-id', shortRunId);
    idNode.title = runId;
    const idCell = el('td');
    const entryWords = isUploadRun(run) ? ['file', 'files'] : ['PDB ID', 'PDB IDs'];
    idCell.append(idNode, el('div', 'run-meta', plural(run.pdbCount || 0, ...entryWords)));
    const row = el('tr');
    row.append(
      idCell,
      statusCell(run),
      progressCell(run),
      el('td', null, formatTimestamp(run.createdAt)),
      el('td', null, sourceLabel(run)),
      actionsCell(run),
    );
    return row;
  }

  // Rebuilding the rows replaces their buttons, so it waits in three cases. While a run action is pending, a
  // listing answered before the action could put back what the action changed. While a pointer is pressed in the
  // table, a click that spans a rebuild would be lost. While a keyboard user is on one of the buttons, their place
  // would be lost. The next listing after that draws the table.
  function runsTableBusy() {
    const focused = document.activeElement;
    return (
      pendingRunActions.size > 0 ||
      pointerDownOnRuns ||
      (runsTableBody.contains(focused) && focused.matches(':focus-visible'))
    );
  }

  function renderRuns(runs) {
    runsSnapshot = runs;
    if (runsTableBusy()) return;
    runsEmpty.hidden = runs.length > 0;
    runsTable.hidden = runs.length === 0;
    runsTableBody.replaceChildren(...runs.map(runRow));
  }

  // The event handlers below respond to the user and to the server's answers. init() at the bottom wires them up.

  function dropSelection() {
    if (lastSelection === null) return;
    lastSelection = null;
    lastSelectionText = '';
    setHint(
      selectHint,
      'The search record was dropped because the technique or X-ray source changed. The identifiers stay in the ' +
        'box and are built as a typed list.',
    );
  }

  async function selectFromPdb() {
    searchFilterMenu.open = false;
    const filterProblem = searchFilterProblem();
    if (filterProblem) {
      setHint(selectHint, filterProblem, true);
      return;
    }
    selectButton.disabled = true;
    setHint(selectHint, 'Searching the PDB…');
    try {
      const data = await fetchJson(
        urls.select,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            technique: techniqueSelect.value,
            xraySource: currentXraySource(),
            ...collectSearchFilters(),
          }),
        },
        'The PDB search failed.',
      );
      const selection = data.selection || null;
      const ids = selection && Array.isArray(selection.pdb_ids) ? selection.pdb_ids : [];
      pdbIdsInput.value = ids.join(', ');
      lastSelection = selection;
      lastSelectionText = normalizeIdText(pdbIdsInput.value);
      validatePdbIds();

      let message;
      if (!data.totalCount) {
        message = `No current PDB entries match ${data.description}.`;
      } else {
        message =
          `${plural(data.totalCount, 'entry matches', 'entries match')} ${data.description} ` +
          `(PDB search at ${data.retrievedAtIso}).`;
        if (!data.truncated) {
          message += ' All loaded.';
        } else if (data.maxPdbIds) {
          message +=
            ` Loaded the first ${Number(data.returnedCount).toLocaleString()}, because this server accepts ` +
            `at most ${Number(data.maxPdbIds).toLocaleString()} IDs per run (MAX_PDB_IDS).`;
        } else {
          message += ` Loaded the first ${Number(data.returnedCount).toLocaleString()}.`;
        }
      }
      (data.warnings || []).forEach((warning) => {
        message += ` ${warning}`;
      });
      setHint(selectHint, message);
    } catch (error) {
      lastSelection = null;
      lastSelectionText = '';
      setHint(selectHint, error.message, true);
    } finally {
      selectButton.disabled = false;
    }
  }

  function handleSearchFilterKeydown(event) {
    // Enter in a range filter runs the search it belongs to, not the build that the form would submit. In the
    // accession list it starts a new line, as in any text area.
    if (event.key !== 'Enter' || event.currentTarget.tagName === 'TEXTAREA') return;
    event.preventDefault();
    if (!selectButton.disabled) selectFromPdb();
  }

  function removeSearchFilter(event) {
    const key = event.currentTarget.dataset.searchFilterRemove;
    searchFilterToggles.find((box) => box.dataset.searchFilterToggle === key).checked = false;
    syncSearchFilters();
    searchFilterMenu.querySelector('summary').focus();
  }

  function closeSearchFilterMenuFromOutside(event) {
    if (searchFilterMenu.open && !searchFilterMenu.contains(event.target)) searchFilterMenu.open = false;
  }

  function handleSearchFilterMenuKeydown(event) {
    if (event.key !== 'Escape' || !searchFilterMenu.open) return;
    searchFilterMenu.open = false;
    searchFilterMenu.querySelector('summary').focus();
  }

  function toggleAllEngines() {
    const available = availableEngineCheckboxes();
    const checked = !allChecked(available);
    // At least one checkbox changes, and its change event updates the button's label.
    available.forEach((checkbox) => setChecked(checkbox, checked));
  }

  function handleFlagChange() {
    updateRefinements();
    updateEngineSelectAllLabel();
  }

  function handleTechniqueChange() {
    applyTechniqueCompatibility();
    applyXraySourceAvailability();
    dropSelection();
  }

  function handleProfileChange() {
    updateProfileDescription();
    applyProfile(profileSelect.value);
    columnFilterDetails.open = true;
  }

  function addExampleId(event) {
    const value = event.currentTarget.dataset.example;
    const current = pdbIdsInput.value.trim();
    pdbIdsInput.value = current ? `${current}, ${value}` : value;
    validatePdbIds();
    pdbIdsInput.focus();
  }

  function handleDragOver(event) {
    event.preventDefault();
    dropzone.classList.add('is-dragover');
  }

  function handleDragLeave(event) {
    event.preventDefault();
    dropzone.classList.remove('is-dragover');
  }

  function handleDrop(event) {
    handleDragLeave(event);
    const dropped = event.dataTransfer ? Array.from(event.dataTransfer.files) : [];
    if (!dropped.length) return;
    const transfer = new DataTransfer();
    dropped.filter((file) => isAcceptedUpload(file.name)).forEach((file) => transfer.items.add(file));
    mmcifFilesInput.files = transfer.files;
    renderFileList();
  }

  function clearBuildFormMarks() {
    const controls = [columnFilterModeSelect, columnFilterListInput, ...countInputs, ...searchFilterInputs];
    controls.forEach((control) => setInvalid(control, false));
  }

  // Return the message that stops a build before it is sent, and mark the control at fault. Return '' when the
  // form may be sent.
  function buildFormProblem(ids, files) {
    clearBuildFormMarks();
    if (!validatePdbIds()) {
      pdbIdsInput.focus();
      return 'Correct the PDB identifiers before building.';
    }
    if (!ids.length && !files.length) return 'Provide PDB IDs or upload mmCIF files before building.';
    if (ids.length && files.length) return 'Choose either PDB IDs or uploaded mmCIF files, not both.';
    const rejected = files.filter((file) => !isAcceptedUpload(file.name)).map((file) => file.name);
    if (rejected.length) {
      const suffixes = config.acceptedSuffixes.join(', ');
      return `Not an mmCIF file: ${rejected.join(', ')}. A file name must end in one of ${suffixes}.`;
    }
    const filterProblem = columnFilterProblem(columnFilterModeSelect, columnFilterListInput);
    if (filterProblem) {
      columnFilterDetails.open = true;
      setInvalid(filterProblem.control, true);
      filterProblem.control.focus();
      return filterProblem.message;
    }
    // A number input whose text is not a number reports an empty value. The form would then silently send no
    // count.
    const badCount = countInputs.find((input) => !input.disabled && input.validity.badInput);
    if (badCount) {
      enginesDetails.open = true;
      setInvalid(badCount, true);
      badCount.focus();
      return 'Enter a count as a positive whole number, or leave it blank for the default.';
    }
    return '';
  }

  async function submitBuild(event) {
    event.preventDefault();
    const ids = splitIds(pdbIdsInput.value);
    const files = Array.from(mmcifFilesInput.files);
    const problem = buildFormProblem(ids, files);
    if (problem) {
      showBuildError(problem);
      return;
    }

    showBuildStatus('Starting the dataset build…');
    buildResultsSection.hidden = true;
    showProgress();
    buildSubmitButton.disabled = true;
    try {
      const settings = buildSettings();
      let request;
      if (files.length) {
        const formData = new FormData();
        files.forEach((file) => formData.append('mmcifFiles', file));
        Object.entries(settings).forEach(([key, value]) => formData.append(key, String(value)));
        request = { method: 'POST', body: formData };
      } else {
        const payload = { pdbIds: ids, ...settings };
        if (lastSelection && normalizeIdText(pdbIdsInput.value) === lastSelectionText) {
          payload.selection = lastSelection;
        }
        request = {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        };
      }
      const data = await fetchJson(
        files.length ? urls.buildUpload : urls.build,
        request,
        'The build could not be started.',
      );
      if (!data.runId) throw new Error('The build could not be started. The server returned no run ID.');
      followRun(data.runId, files.length > 0);
      refreshRuns();
    } catch (error) {
      showBuildError(error.message);
    } finally {
      buildSubmitButton.disabled = false;
    }
  }

  function staleMessage(fromUpload) {
    const next = fromUpload
      ? 'Upload the files again to start a new run.'
      : 'Resume it from the Runs panel to build the entries that are left.';
    return `This run stopped before it finished, as when the server restarts. ${next}`;
  }

  // Draw a status poll's answer and return whether the run is still building.
  function renderStatus(data, runId, fromUpload) {
    if (data.status === 'running') {
      updateProgress(data.progress);
      return true;
    }
    statusProgress.hidden = true;
    if (data.status === 'completed') {
      showBuildStatus('Dataset ready.');
      if (data.result) renderBuildResults(data.result, runId);
    } else if (data.status === 'paused' && data.result) {
      showBuildStatus('Run paused. The partial dataset below holds the entries built so far.');
      renderBuildResults(data.result, runId);
    } else if (data.status === 'paused') {
      showBuildStatus('Run paused before any entry was built, so there is no partial dataset.');
    } else if (data.status === 'error') {
      showBuildError(data.error || 'The dataset build failed.');
    } else if (data.status === 'stale') {
      showBuildError(staleMessage(fromUpload));
    } else {
      showBuildError(`Run ${runId} has no status the page can show.`);
    }
    return false;
  }

  function stopFollowing() {
    statusPollGeneration += 1;
    clearTimeout(statusPollTimer);
    followedRunId = null;
  }

  function followRun(runId, fromUpload) {
    stopFollowing();
    const generation = statusPollGeneration;
    followedRunId = runId;
    const url = expandTemplate(urls.runStatus, runId);
    let failures = 0;

    const poll = async () => {
      let data;
      try {
        data = await fetchJson(url, undefined, 'The run status could not be read.');
      } catch (error) {
        if (generation !== statusPollGeneration) return;
        // No answer, or a server error, may pass with time. A 404 or another refusal will not.
        const transient = error.status === 0 || error.status >= 500;
        if (transient && failures < STATUS_POLL_RETRIES) {
          failures += 1;
          statusPollTimer = setTimeout(poll, STATUS_POLL_MILLISECONDS * 2 ** failures);
          return;
        }
        stopFollowing();
        showBuildError(error.message);
        return;
      }
      if (generation !== statusPollGeneration) return;
      failures = 0;
      if (renderStatus(data, runId, fromUpload)) {
        statusPollTimer = setTimeout(poll, STATUS_POLL_MILLISECONDS);
      } else {
        stopFollowing();
        refreshRuns();
      }
    };
    poll();
  }

  function scheduleRunsPoll() {
    clearTimeout(runsPollTimer);
    // A hidden tab polls nothing, and neither does a switched-off auto-refresh. Showing the tab, or switching
    // auto-refresh on, refreshes the panel.
    if (document.hidden || !runsAutoRefreshCheckbox.checked) return;
    const delay = anyRunLive() ? RUNS_POLL_LIVE_MILLISECONDS : RUNS_POLL_IDLE_MILLISECONDS;
    runsPollTimer = setTimeout(refreshRuns, delay);
  }

  async function refreshRuns() {
    // One listing is in flight at a time. A refresh asked for meanwhile runs once that listing answers. A listing
    // sent before a run action therefore cannot be the last one drawn.
    if (runsRequestInFlight) {
      runsRefreshQueued = true;
      return;
    }
    runsRequestInFlight = true;
    clearTimeout(runsPollTimer);
    try {
      const data = await fetchJson(urls.runs, undefined, 'The runs could not be listed.');
      renderRuns(Array.isArray(data.runs) ? data.runs : []);
      if (runsStatusShowsListingError) runsStatus.hidden = true;
    } catch (error) {
      setRunsStatus(error.message, true);
      runsStatusShowsListingError = true;
    } finally {
      runsRequestInFlight = false;
      if (runsRefreshQueued) {
        runsRefreshQueued = false;
        refreshRuns();
      } else {
        scheduleRunsPoll();
      }
    }
  }

  async function callRunAction(action, runId) {
    const { template, method, verb } = RUN_ACTIONS[action];
    await fetchJson(expandTemplate(template, runId), { method }, `Run ${runId} could not be ${verb}.`);
    if (action === 'pause') {
      setRunsStatus(`Pausing run ${runId}.`);
    } else if (action === 'resume') {
      setRunsStatus(`Resumed run ${runId}. Its progress is shown under the build form.`);
      showBuildStatus(`Resuming run ${runId}…`);
      buildResultsSection.hidden = true;
      showProgress();
      followRun(runId, false);
    } else {
      setRunsStatus(`Deleted run ${runId}.`);
    }
  }

  async function handleRunsTableClick(event) {
    const button = event.target.closest('button[data-action]');
    if (!button) return;
    const { action, runId } = button.dataset;
    if (pendingRunActions.has(runId)) return;
    if (action === 'delete' && !confirm(`Delete run ${runId}? This removes its partial files and dataset.`)) {
      return;
    }

    pendingRunActions.add(runId);
    runsTableBody.querySelectorAll('button[data-run-id]').forEach((other) => {
      if (other.dataset.runId === runId) other.disabled = true;
    });
    try {
      await callRunAction(action, runId);
    } catch (error) {
      setRunsStatus(error.message, true);
    } finally {
      pendingRunActions.delete(runId);
      refreshRuns();
    }
  }

  async function submitFilter(event) {
    event.preventDefault();
    setInvalid(filterModeSelect, false);
    setInvalid(filterColumnsInput, false);
    if (!filterFileInput.files.length) {
      setStatus(filterStatus, 'Select a CSV file to filter.', true);
      filterFileInput.focus();
      return;
    }
    const filterProblem = columnFilterProblem(filterModeSelect, filterColumnsInput);
    if (filterProblem) {
      setInvalid(filterProblem.control, true);
      filterProblem.control.focus();
      setStatus(filterStatus, filterProblem.message, true);
      return;
    }

    setStatus(filterStatus, 'Filtering the dataset…');
    filterResultsSection.hidden = true;
    const formData = new FormData();
    formData.append('file', filterFileInput.files[0]);
    formData.append('columnFilterMode', filterModeSelect.value);
    formData.append('columnFilterList', filterColumnsInput.value);
    filterSubmitButton.disabled = true;
    try {
      const data = await fetchJson(
        urls.filterUpload,
        { method: 'POST', body: formData },
        'The dataset could not be filtered.',
      );
      setStatus(filterStatus, 'Filtered dataset ready.');
      renderFilterResults(data);
    } catch (error) {
      setStatus(filterStatus, error.message, true);
    } finally {
      filterSubmitButton.disabled = false;
    }
  }

  function handleBuildReset() {
    // The reset event fires before the controls get their default values back. The page is therefore brought in
    // line with them once the reset has happened.
    setTimeout(() => {
      // A build keeps running when the form is cleared, so the status panel keeps following it.
      if (followedRunId === null) {
        statusPanel.hidden = true;
        buildResultsSection.hidden = true;
      }
      lastSelection = null;
      lastSelectionText = '';
      appliedProfile = null;
      setHint(selectHint);
      validatePdbIds();
      clearBuildFormMarks();
      searchFilterMenu.open = false;
      syncSearchFilters();
      updateProfileDescription();
      renderFileList();
      applyTechniqueCompatibility();
      applyXraySourceAvailability();
    }, 0);
  }

  function handleFilterReset() {
    setInvalid(filterModeSelect, false);
    setInvalid(filterColumnsInput, false);
    filterStatus.hidden = true;
    filterResultsSection.hidden = true;
  }

  function handleVisibilityChange() {
    if (!document.hidden) refreshRuns();
  }

  function handleAutoRefreshChange() {
    if (runsAutoRefreshCheckbox.checked) {
      refreshRuns();
    } else {
      clearTimeout(runsPollTimer);
    }
  }

  function init() {
    buildForm.addEventListener('submit', submitBuild);
    buildForm.addEventListener('reset', handleBuildReset);
    pdbIdsInput.addEventListener('input', validatePdbIds);
    document.querySelectorAll('.chip[data-example]').forEach((chip) => {
      chip.addEventListener('click', addExampleId);
    });
    selectButton.addEventListener('click', selectFromPdb);
    searchFilterInputs.forEach((input) => {
      input.addEventListener('input', () => setInvalid(input, false));
      input.addEventListener('keydown', handleSearchFilterKeydown);
    });
    searchFilterToggles.forEach((box) => box.addEventListener('change', syncSearchFilters));
    document.querySelectorAll('[data-search-filter-remove]').forEach((button) => {
      button.addEventListener('click', removeSearchFilter);
    });
    searchFilterMenu.addEventListener('keydown', handleSearchFilterMenuKeydown);
    document.addEventListener('click', closeSearchFilterMenuFromOutside);
    mmcifFilesInput.addEventListener('change', renderFileList);
    ['dragenter', 'dragover'].forEach((type) => dropzone.addEventListener(type, handleDragOver));
    dropzone.addEventListener('dragleave', handleDragLeave);
    dropzone.addEventListener('drop', handleDrop);
    techniqueSelect.addEventListener('change', handleTechniqueChange);
    xraySourceSelect.addEventListener('change', dropSelection);
    profileSelect.addEventListener('change', handleProfileChange);
    columnFilterModeSelect.addEventListener('change', () => setInvalid(columnFilterModeSelect, false));
    columnFilterListInput.addEventListener('input', () => setInvalid(columnFilterListInput, false));
    countInputs.forEach((input) => input.addEventListener('input', () => setInvalid(input, false)));
    flagInputs.forEach((checkbox) => checkbox.addEventListener('change', handleFlagChange));
    engineSelectAllButton.addEventListener('click', toggleAllEngines);

    filterForm.addEventListener('submit', submitFilter);
    filterForm.addEventListener('reset', handleFilterReset);

    runsRefreshButton.addEventListener('click', refreshRuns);
    runsAutoRefreshCheckbox.addEventListener('change', handleAutoRefreshChange);
    runsTableBody.addEventListener('click', handleRunsTableClick);
    runsTableBody.addEventListener('pointerdown', () => {
      pointerDownOnRuns = true;
    });
    ['pointerup', 'pointercancel'].forEach((type) =>
      document.addEventListener(type, () => {
        pointerDownOnRuns = false;
      }),
    );
    document.addEventListener('visibilitychange', handleVisibilityChange);

    applyTechniqueCompatibility();
    applyXraySourceAvailability();
    // A browser may restore the checkboxes it held before a reload, so the fields are brought in line with them.
    syncSearchFilters();
    updateProfileDescription();
    renderFileList();
    refreshRuns();
  }

  init();
})();
