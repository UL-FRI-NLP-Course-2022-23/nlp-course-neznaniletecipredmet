document.getElementById("myForm").addEventListener("submit", function(event) {
    event.preventDefault(); // Prevent form submission

    // Get the text input value
    var textInput = document.getElementById("textInput").value;

    // Send an AJAX request to the server
    var xhr = new XMLHttpRequest();
    xhr.open("POST", "/process", true);
    xhr.setRequestHeader("Content-Type", "application/json");
    xhr.onreadystatechange = function() {
        if (xhr.readyState === 4 && xhr.status === 200) {
            console.log("Response received: " + xhr.responseText);
//            location.reload ? location.reload() : location = location;
        }
    };
    xhr.send(JSON.stringify({ text: textInput }));
});

document.addEventListener("DOMContentLoaded", function() {
  var sketchSelector = document.getElementById("sketchSelector");

  // Send an AJAX request to retrieve the file names from the server
  var xhr = new XMLHttpRequest();
  xhr.open("GET", "/getFiles", true);
  xhr.onreadystatechange = function() {
    if (xhr.readyState === 4 && xhr.status === 200) {
      var fileNames = JSON.parse(xhr.responseText);
      addOptionsToSelect(fileNames);
    }
  };
  xhr.send();
});

function addOptionsToSelect(fileNames) {
  var sketchSelector = document.getElementById("sketchSelector");

  fileNames.forEach(function(fileName) {
    var optionElement = document.createElement("option");
    optionElement.value = fileName;
    optionElement.textContent = fileName;
    sketchSelector.appendChild(optionElement);
  });
}