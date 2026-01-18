import { h, render } from "preact";

import { App } from "App";

window.addEventListener("load", function() {
	render(<App />, document.querySelector("body")!);
});