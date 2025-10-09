// Autocomplete for player search inputs (home & showdown)
(() => {
	const endpoints = {
		search: '/api/search'
	};

	function debounce(fn, wait) {
		let t = null;
		return (...args) => {
			clearTimeout(t);
			t = setTimeout(() => fn(...args), wait);
		};
	}

	async function fetchSuggestions(q) {
		if (!q || q.trim().length === 0) return [];
		try {
			const res = await fetch(`${endpoints.search}?q=${encodeURIComponent(q)}`);
			if (!res.ok) return [];
			const json = await res.json();
			return json.players || [];
		} catch (e) {
			console.error('fetchSuggestions error', e);
			return [];
		}
	}

	function renderSuggestions(container, items) {
		container.innerHTML = '';
		if (!items || items.length === 0) {
			container.classList.remove('show');
			return;
		}

		items.forEach((it, idx) => {
			const div = document.createElement('div');
			div.className = 'suggestion-item';
			div.textContent = it;
			div.dataset.index = idx;
			container.appendChild(div);
		});
		container.classList.add('show');
	}

	function attachAutocomplete(inputEl, suggestionsEl) {
		let items = [];
		let selected = -1;

		const debounced = debounce(async (val) => {
			items = await fetchSuggestions(val);
			selected = -1;
			renderSuggestions(suggestionsEl, items);
		}, 250);

		inputEl.addEventListener('input', (e) => {
			const val = e.target.value;
			debounced(val);
		});

		inputEl.addEventListener('keydown', (e) => {
			const children = Array.from(suggestionsEl.children);
			if (e.key === 'ArrowDown') {
				e.preventDefault();
				selected = Math.min(selected + 1, children.length - 1);
				children.forEach(c => c.classList.remove('selected'));
				if (children[selected]) children[selected].classList.add('selected');
			} else if (e.key === 'ArrowUp') {
				e.preventDefault();
				selected = Math.max(selected - 1, 0);
				children.forEach(c => c.classList.remove('selected'));
				if (children[selected]) children[selected].classList.add('selected');
			} else if (e.key === 'Enter') {
				if (selected >= 0 && children[selected]) {
					e.preventDefault();
					const name = children[selected].dataset.name || children[selected].textContent;
					inputEl.value = name;
					suggestionsEl.classList.remove('show');
					if (typeof window.selectPlayer === 'function') {
						window.selectPlayer(name);
						return;
					}
					inputEl.dispatchEvent(new Event('change', { bubbles: true }));
				}
			} else if (e.key === 'Escape') {
				suggestionsEl.classList.remove('show');
			}
		});

		suggestionsEl.addEventListener('click', (e) => {
			const it = e.target.closest('.suggestion-item');
			if (!it) return;
			const name = it.dataset.name || it.textContent;
			inputEl.value = name;
			suggestionsEl.classList.remove('show');
			// If page defines global selectPlayer, call it to load immediately
			if (typeof window.selectPlayer === 'function') {
				window.selectPlayer(name);
				return;
			}
			inputEl.dispatchEvent(new Event('change', { bubbles: true }));
		});

		document.addEventListener('click', (e) => {
			if (!inputEl.contains(e.target) && !suggestionsEl.contains(e.target)) {
				suggestionsEl.classList.remove('show');
			}
		});
	}

	function init() {
		const mappings = [
			{ input: 'player-search', sugg: 'search-suggestions' },
			{ input: 'player1-search', sugg: 'player1-suggestions' },
			{ input: 'player2-search', sugg: 'player2-suggestions' }
		];

		mappings.forEach(m => {
			const inputEl = document.getElementById(m.input);
			const suggestionsEl = document.getElementById(m.sugg);
			if (inputEl && suggestionsEl) attachAutocomplete(inputEl, suggestionsEl);
		});
	}

	// Auto-init when DOM ready
	if (document.readyState === 'loading') {
		document.addEventListener('DOMContentLoaded', init);
	} else {
		init();
	}
})();