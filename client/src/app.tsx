import React from 'react';
import GlobalStyle from './globalStyle';
import Landing from './components/landing';

const App = () => {
    return (
        <React.Fragment>
            <GlobalStyle />
            <Landing />
        </React.Fragment>
    );
};

export default App;
