import React from 'react';
import GlobalStyle from './globalStyle';

const App = () => {
    return (
        <React.Fragment>
            <div>
                <h1>Pripol</h1>

                <input type='text' placeholder='' />
            </div>

            <GlobalStyle />
        </React.Fragment>
    );
};

export default App;
