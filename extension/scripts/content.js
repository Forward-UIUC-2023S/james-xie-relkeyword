const article = document.querySelector("textarea");
chrome.runtime.sendMessage({message: "Hello from content.js!"});
if (article) {
  article.addEventListener('click', ()=>{

    console.log("Sending Message");
    // var port = chrome.runtime.connect({name: "knockknock"});
    // port.postMessage({name: article.value});
    chrome.runtime.sendMessage({message: "input_word",
                                input_word: article.value});

    });

}
console.log("HEY!");


for(let i = 0; i < document.querySelectorAll('input').length; i++){
  console.log("HEH")
  document.querySelectorAll('input')[i].addEventListener('click', ()=>{
      //Get the DOM element of whatever input user clicks on
      var currentInput = document.querySelectorAll('input')[i];
      
      chrome.runtime.sendMessage({message: "input_word",
                                input_word: currentInput.value});


      //Just for example purposes 
      // document.getElementById('display').innerText = 'You clicked on input...' + (i+1);
  });
}